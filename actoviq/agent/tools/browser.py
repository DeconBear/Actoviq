"""
Browser Automation Tool (Playwright Integration)

Provides a persistent browser session for the Agent to inspect and interact with web pages.
"""

import os
import time
import base64
import queue
import threading
from typing import Dict, Optional, List, Any, Callable

try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext, Route
except ImportError:
    sync_playwright = None
    Page = None
    BrowserContext = None

class BrowserManager:
    """
    Manages a persistent Playwright browser session.
    """
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self._startup_error: Optional[str] = None
        self._startup_event = threading.Event()
        self._startup_timeout = int(os.environ.get("KLYNX_BROWSER_STARTUP_TIMEOUT", "20"))
        
        self._task_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        self._ensure_playwright_installed()

    def _ensure_playwright_installed(self):
        if sync_playwright is None:
            print("[Warning] Playwright not installed. Browser tools will be unavailable.")

    def _worker_loop(self):
        """Dedicated thread to run Playwright to avoid cross-thread loop issues."""
        global sync_playwright, Page, BrowserContext
        if not sync_playwright:
            try:
                from playwright.sync_api import sync_playwright as sp
                sync_playwright = sp
            except ImportError:
                self._startup_error = (
                    "Playwright is not installed. Install dependency with "
                    "`pip install playwright`, then run `playwright install chromium`."
                )
                print(f"[Error] Failed to start browser: {self._startup_error}")
                self._running = False
                self._startup_event.set()
                return

        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(
                 viewport={"width": 1280, "height": 800},
                 user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self.page = self.context.new_page()
            print("[System] Browser started in dedicated worker thread.")
            self._startup_event.set()
        except Exception as e:
            self._startup_error = self._format_startup_error(e)
            print(f"[Error] Failed to start browser: {self._startup_error}")
            self._running = False
            self._startup_event.set()
            return
            
        while self._running:
            try:
                task, result_queue = self._task_queue.get(timeout=0.5)
                try:
                    res = task()
                    result_queue.put({"status": "ok", "result": res})
                except Exception as e:
                    result_queue.put({"status": "error", "error": str(e)})
                finally:
                    self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Warning] Browser worker error: {e}")

        # Cleanup
        try:
            if self.context: self.context.close()
            if self.browser: self.browser.close()
            if self.playwright: self.playwright.stop()
        except:
            pass
        self.context = None
        self.browser = None
        self.playwright = None
        self.page = None
    
    def _format_startup_error(self, exc: Exception) -> str:
        msg = str(exc)
        lowered = msg.lower()
        if "executable doesn't exist" in lowered or "browser_type.launch" in lowered:
            return (
                f"Playwright browser executable is missing or failed to launch: {msg}. "
                "Run `playwright install chromium` and retry."
            )
        return f"Failed to start Playwright browser: {msg}"

    def _execute_in_thread(self, task: Callable) -> Any:
        self.ensure_page()
        if not self._running or self._worker_thread is None or not self._worker_thread.is_alive():
            detail = self._startup_error or "Browser worker thread is not running."
            raise Exception(detail)
        res_queue = queue.Queue()
        self._task_queue.put((task, res_queue))
        try:
            result = res_queue.get(timeout=65)
        except queue.Empty:
            raise Exception("Browser task timed out while waiting for worker response.")
        if result["status"] == "error":
            raise Exception(result["error"])
        return result["result"]

    def start(self):
        """Start the browser session worker thread."""
        if self._running and self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._startup_event.clear()
        self._startup_error = None
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        # Wait for initialization
        started = self._startup_event.wait(timeout=self._startup_timeout)
        if not started:
            self.stop()
            raise Exception(
                f"Timed out waiting for browser to start ({self._startup_timeout}s). "
                "If this machine is slow, set KLYNX_BROWSER_STARTUP_TIMEOUT to a larger value."
            )
        if self._startup_error:
            self.stop()
            raise Exception(self._startup_error)
        if self.page is None:
            self.stop()
            raise Exception("Browser start failed: page initialization did not complete.")

    def stop(self):
        """Stop the browser session."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=3)
        self._worker_thread = None

    def ensure_page(self):
        if not self._running or self.page is None:
            self.start()

    def goto(self, url: str) -> str:
        """Navigate to a URL."""
        def _job():
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            status = response.status if response else "unknown"
            return f"<success>Navigated to {url} (Status: {status})</success>"
            
        try:
            return self._execute_in_thread(_job)
        except Exception as e:
            return f"<error>Navigation failed: {str(e)}</error>"

    def get_content(self, selector: str = None) -> str:
        """Get page content or text of an element."""
        def _job():
            if selector:
                if not self.page.locator(selector).count():
                     return f"<error>Selector '{selector}' not found</error>"
                return self.page.locator(selector).inner_text()
            else:
                title = self.page.title()
                text = self.page.inner_text("body")
                preview = text[:2000] + "..." if len(text) > 2000 else text
                
                interactives = self.page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a, button, input, textarea, select'))
                        .filter(el => {
                            let rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        })
                        .map(el => {
                            let text = (el.innerText || el.value || el.placeholder || '').trim().replace(/\\n/g, ' ');
                            let info = `<${el.tagName.toLowerCase()}`;
                            if(el.id) info += ` id="${el.id}"`;
                            if(el.name) info += ` name="${el.name}"`;
                            if(el.className && typeof el.className === 'string') info += ` class="${el.className}"`;
                            info += `>`;
                            if(text) info += text.substring(0, 50);
                            return info;
                        }).slice(0, 100);
                }''')
                interactives_text = "\n".join(interactives)
                return f"Page: {title} ({self.page.url})\n\n[Interactive Elements (Available for `browser_act`)]:\n{interactives_text}\n\n[Content Preview]:\n{preview}"

        try:
            return self._execute_in_thread(_job)
        except Exception as e:
            return f"<error>Get content failed: {str(e)}</error>"

    def act(self, action: str, selector: str, value: str = None) -> str:
        """Perform an action on the page."""
        def _job():
            target = self.page.locator(selector)
            if not target.count():
                 return f"<error>Element '{selector}' not found</error>"

            if action == "click":
                target.first.click()
                return f"<success>Clicked {selector}</success>"
            elif action == "press":
                if value is None:
                    return "<error>Value required for press action (e.g. 'Enter')</error>"
                target.first.press(value)
                return f"<success>Pressed '{value}' on {selector}</success>"
            elif action == "type":
                if value is None:
                    return "<error>Value required for type action</error>"
                target.first.fill(value)
                return f"<success>Typed '{value}' into {selector}</success>"
            elif action == "hover":
                target.first.hover()
                return f"<success>Hovered over {selector}</success>"
            else:
                return f"<error>Unknown action: {action}</error>"
                
        try:
            return self._execute_in_thread(_job)
        except Exception as e:
            return f"<error>Action failed: {str(e)}</error>"

    def scroll(self, direction: str = "down", amount: int = None) -> str:
        """Scroll the page."""
        def _job():
            if amount is not None:
                scroll_y = int(amount) if direction == "down" else -int(amount)
            else:
                scroll_y = 600 if direction == "down" else -600
                
            self.page.evaluate(f"window.scrollBy(0, {scroll_y})")
            # Wait a bit for lazy-loaded content or smooth scroll to finish
            self.page.wait_for_timeout(500)
            return f"<success>Scrolled {direction} by {abs(scroll_y)} pixels</success>"
            
        try:
            return self._execute_in_thread(_job)
        except Exception as e:
            return f"<error>Scroll failed: {str(e)}</error>"

    def screenshot(self) -> str:
        """Take a screenshot and return the path."""
        def _job():
            filename = f"screenshot_{int(time.time())}.png"
            path = os.path.abspath(filename)
            self.page.screenshot(path=path)
            return f"<success>Screenshot saved to {path}</success>"
            
        try:
            return self._execute_in_thread(_job)
        except Exception as e:
            return f"<error>Screenshot failed: {str(e)}</error>"

    def get_console_logs(self) -> str:
        # Implementing log capture requires event listeners.
        # For simplicity, we might skip this or implement a log buffer.
        return "<info>Console logs not implemented yet</info>"
