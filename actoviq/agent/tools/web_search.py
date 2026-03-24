"""Tavily web search integration."""

import os
from typing import Optional

_TAVILY_API_KEY: Optional[str] = None


def set_tavily_api(api_key: str):
    """Set Tavily API key in module scope."""
    global _TAVILY_API_KEY
    _TAVILY_API_KEY = str(api_key or "").strip() or None


def resolve_tavily_api_key(explicit_api_key: Optional[str] = None) -> str:
    """Resolve Tavily key with priority: explicit arg -> module key -> env."""
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit
    if _TAVILY_API_KEY:
        return str(_TAVILY_API_KEY).strip()
    return str(os.getenv("TAVILY_API_KEY", "") or "").strip()


def is_tavily_configured(explicit_api_key: Optional[str] = None) -> bool:
    """Whether Tavily API key is currently available."""
    return bool(resolve_tavily_api_key(explicit_api_key))


class WebSearchTool:
    """Tavily web search tool."""

    def __init__(self, api_key: Optional[str] = None):
        self._explicit_api_key = str(api_key or "").strip()
        self.api_key = resolve_tavily_api_key(self._explicit_api_key)
        self._client = None

    def _refresh_api_key(self) -> str:
        resolved = resolve_tavily_api_key(self._explicit_api_key)
        if resolved != self.api_key:
            self.api_key = resolved
            self._client = None
        return self.api_key

    def _get_client(self):
        """Create Tavily client lazily."""
        api_key = self._refresh_api_key()
        if self._client is None:
            if not api_key:
                raise RuntimeError(
                    "Tavily API key is not configured. Use set_tavily_api(...) "
                    "or set TAVILY_API_KEY in environment."
                )
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=api_key)
        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> str:
        """Run Tavily search and return XML output."""
        try:
            client = self._get_client()

            proxy_vars = {}
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                if key in os.environ:
                    proxy_vars[key] = os.environ.pop(key)
            old_no_proxy = os.environ.get("NO_PROXY")
            os.environ["NO_PROXY"] = "*"

            try:
                result = client.search(
                    query=query,
                    max_results=min(int(max_results), 10),
                    search_depth=search_depth,
                    include_answer=include_answer,
                )
            finally:
                os.environ.update(proxy_vars)
                if old_no_proxy is not None:
                    os.environ["NO_PROXY"] = old_no_proxy
                elif "NO_PROXY" in os.environ:
                    del os.environ["NO_PROXY"]

            xml = [f'<search_results query="{self._escape_xml(query)}">']
            if include_answer and result.get("answer"):
                xml.append(f'  <answer>{self._escape_xml(result["answer"])}</answer>')
            for i, item in enumerate(result.get("results", []), 1):
                title = self._escape_xml(item.get("title", ""))
                url = self._escape_xml(item.get("url", ""))
                content = self._escape_xml(item.get("content", ""))
                score = item.get("score", 0)
                xml.append(f'  <result rank="{i}" score="{score:.2f}">')
                xml.append(f"    <title>{title}</title>")
                xml.append(f"    <url>{url}</url>")
                xml.append(f"    <content>{content}</content>")
                xml.append("  </result>")

            xml.append("</search_results>")
            return "\n".join(xml)
        except RuntimeError:
            raise
        except Exception as e:
            return f"<error>: {str(e)}</error>"

    @staticmethod
    def _escape_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
