
import json
import os
import subprocess
import threading
import time
import queue
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSPClient")

class LSPClient:
    def __init__(self, command: List[str] = None):
        if command is None:
            # Default to running via python -m to avoid path/shell issues
            import sys
            self.command = [sys.executable, "-m", "pylsp"]
        else:
            self.command = command
            
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.request_id = 0
        self.responses: Dict[int, queue.Queue] = {}
        self.diagnostics: Dict[str, List[Dict]] = {}
        self.server_capabilities = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def start(self):
        """Start the LSP server subprocess."""
        try:
            # Use shell=True specifically for Windows if needed, or better, ensure executable is in path
            # For strict subprocess usage, we often avoid shell=True unless necessary.
            # On Windows, "pylsp" might be a script.
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True if os.name == 'nt' else False # Often needed on Windows for entry point scripts
            )
            self.running = True
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()
            
            # Start stderr reader thread for debugging
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()
            
            logger.info(f"LSP server started: {self.command}")
            
        except Exception as e:
            logger.error(f"Failed to start LSP server: {e}")
            raise

    def stop(self):
        """Stop the LSP server."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            logger.info("LSP server stopped")

    def _read_stderr(self):
        """Read stderr from the server for logging."""
        while self.running and self.process:
            line = self.process.stderr.readline()
            if not line:
                break
            # logger.debug(f"LSP STDERR: {line.decode('utf-8').strip()}")

    def _read_loop(self):
        """Read stdout from the server (JSON-RPC)."""
        while self.running and self.process:
            try:
                # Read Content-Length header
                header = b""
                while True:
                    byte = self.process.stdout.read(1)
                    if not byte:
                        # End of stream
                        self.running = False
                        return
                    header += byte
                    if header.endswith(b"\r\n\r\n"):
                        break
                
                # Parse Content-Length
                content_length = 0
                for line in header.decode("ascii").split("\r\n"):
                    if line.startswith("Content-Length:"):
                        content_length = int(line.split(":")[1].strip())
                
                if content_length > 0:
                    content = self.process.stdout.read(content_length)
                    message = json.loads(content.decode("utf-8"))
                    self._handle_message(message)
                    
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                import traceback
                traceback.print_exc()
                if not self.process or self.process.poll() is not None:
                    self.running = False
                    break

    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming JSON-RPC message."""
        if "id" in message and "method" not in message:
            # Response to a request
            req_id = message["id"]
            if req_id in self.responses:
                self.responses[req_id].put(message)
        elif "method" in message:
            # Notification or Request from server
            method = message["method"]
            params = message.get("params", {})
            
            if method == "textDocument/publishDiagnostics":
                self._handle_diagnostics(params)
            # Add other notifications as needed

    def _handle_diagnostics(self, params: Dict[str, Any]):
        """Store diagnostics for a file."""
        uri = params["uri"]
        diagnostics = params["diagnostics"]
        # Convert URI to path for easier access? Or keep as URI.
        # Let's keep as URI in internal storage but maybe provide helper.
        self.diagnostics[uri] = diagnostics
        logger.info(f"Received {len(diagnostics)} diagnostics for {uri}")

    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a request and wait for the response."""
        with self._response_lock:
            self.request_id += 1
            req_id = self.request_id
            self.responses[req_id] = queue.Queue()

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {}
        }
        
        self._send_json(request)
        
        try:
            # Wait for response (timeout 10s)
            response = self.responses[req_id].get(timeout=10)
            del self.responses[req_id]
            
            if "error" in response:
                raise Exception(f"LSP Error: {response['error']}")
            
            return response.get("result")
        except queue.Empty:
            del self.responses[req_id]
            raise TimeoutError(f"Request {method} timed out")

    def send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send a notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        self._send_json(notification)

    def _send_json(self, data: Dict[str, Any]):
        """Send JSON data to the server."""
        content = json.dumps(data).encode("utf-8")
        header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
        
        with self._write_lock:
            if self.process and self.process.stdin:
                self.process.stdin.write(header + content)
                self.process.stdin.flush()

    # --- High-level methods ---

    def initialize(self, root_path: str):
        """Initialize the LSP session."""
        root_uri = Path(root_path).as_uri()
        params = {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {},
                    "synchronization": {
                        "didSave": True
                    },
                    "completion": {},
                    "definition": {}
                }
            }
        }
        result = self.send_request("initialize", params)
        self.server_capabilities = result.get("capabilities", {})
        self.send_notification("initialized", {})
        return result

    def open_document(self, file_path: str):
        """Send textDocument/didOpen notification."""
        path = Path(file_path).resolve()
        uri = path.as_uri()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return

        params = {
            "textDocument": {
                "uri": uri,
                "languageId": "python", # Assuming python for now, can extract extension
                "version": 1,
                "text": text
            }
        }
        self.send_notification("textDocument/didOpen", params)
        # Assuming version 1 for now. If we track changes, we need to increment.

    def request_definition(self, file_path: str, line: int, character: int):
        """Request definition location."""
        # Line and character are 0-based in LSP
        path = Path(file_path).resolve()
        uri = path.as_uri()
        
        params = {
            "textDocument": {
                "uri": uri
            },
            "position": {
                "line": line,
                "character": character
            }
        }
        return self.send_request("textDocument/definition", params)

    def get_file_diagnostics(self, file_path: str) -> List[Dict]:
        """Get diagnostics for a specific file."""
        path = Path(file_path).resolve()
        uri = path.as_uri()
        return self.diagnostics.get(uri, [])

