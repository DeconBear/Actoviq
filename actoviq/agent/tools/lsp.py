
import os
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from .lsp_client import LSPClient

# Global LSP Client instance
_lsp_client: Optional[LSPClient] = None

def init_lsp(root_path: str = ".") -> str:
    """Initialize the LSP client if not already running."""
    global _lsp_client
    try:
        # Use default command logic in LSPClient (which uses sys.executable -m pylsp)
        _lsp_client = LSPClient()
        _lsp_client.start()
        
        # Initialize
        # Using abspath for root_path
        abs_root = os.path.abspath(root_path)
        result = _lsp_client.initialize(abs_root)
        
        return f"<success>LSP Client initialized for root: {abs_root}</success>"
    except Exception as e:
        return f"<error>Failed to initialize LSP Client: {e}</error>"

def shutdown_lsp() -> str:
    """Stop the LSP client."""
    global _lsp_client
    if _lsp_client:
        _lsp_client.stop()
        _lsp_client = None
        return "<success>LSP Client stopped.</success>"
    return "<info>LSP Client was not running.</info>"

def _ensure_lsp():
    """Ensure LSP client is running and initialized."""
    global _lsp_client
    if not _lsp_client or not _lsp_client.running:
        # Auto-initialize with current working directory
        init_lsp(os.getcwd())
        # Give it a moment to start up
        time.sleep(1)

def get_diagnostics(file_path: str) -> str:
    """
    Get diagnostics (errors/warnings) for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Formatted string string of diagnostics
    """
    _ensure_lsp()
    if not _lsp_client or not _lsp_client.running:
         return "<error>LSP Client failed to start.</error>"

    abs_path = os.path.abspath(file_path)
    
    # Open the document to get fresh diagnostics
    # Note: pylsp might publish diagnostics asynchronously after didOpen or didChange
    _lsp_client.open_document(abs_path)
    
    # Wait briefly for diagnostics to arrive (naive approach)
    # In a real async system we'd listen for the notification. 
    # Here we poll for a short duration.
    for _ in range(10): # Wait up to 2 seconds
        # We need check if we HAVE diagnostics for this file
        # But this is tricky because we might have old ones.
        # For now, let's just wait a fixed time and check.
        time.sleep(0.2)
        
    diags = _lsp_client.get_file_diagnostics(abs_path)
    
    if not diags:
        return f"<info>No diagnostics found for {file_path}. (Either no errors, or analysis not finished)</info>"
    
    # Format diagnostics
    output = []
    for d in diags:
        severity_map = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}
        severity = severity_map.get(d.get("severity", 1), "Unknown")
        message = d.get("message", "")
        line = d.get("range", {}).get("start", {}).get("line", 0) + 1
        output.append(f"Line {line}: [{severity}] {message}")
        
    return "\n".join(output)

def goto_definition(file_path: str, line: int, character: int) -> str:
    """
    Go to definition of symbol at line, character.
    
    Args:
        file_path: Path to file
        line: Line number (1-based)
        character: Character column (1-based)
        
    Returns:
        Location of definition
    """
    _ensure_lsp()
    if not _lsp_client or not _lsp_client.running:
         return "<error>LSP Client failed to start.</error>"

    abs_path = os.path.abspath(file_path)
    _lsp_client.open_document(abs_path)
    
    # Convert to 0-based for LSP
    lsp_line = line - 1
    lsp_char = character - 1
    
    try:
        # Request definition
        # The client method expects send_request/send_notification logic
        # We need to implement request logic in LSPClient properly
        # which we did in previous step (request_definition)
        
        # Wait for response logic is in send_request
        # We need to call request_definition on _lsp_client instance
        
        # NOTE: logic in lsp_client.py: request_definition(self, file_path, line, character)
        # It calls send_request("textDocument/definition", ...)
        # send_request waits for response.
        
        result = _lsp_client.request_definition(abs_path, lsp_line, lsp_char) # Passing path, not uri?
        # Ah wait, request_definition in lsp_client.py takes file_path string. Correct.
        
        if not result:
            return "<info>No definition found.</info>"
            
        # Result can be Location or Location[] or LocationLink[]
        # Simplify handling
        locations = result if isinstance(result, list) else [result]
        
        output = []
        for loc in locations:
            uri = loc.get("uri", "")
            if not uri: continue
            
            # Convert URI to path
            # naive: file:///path -> /path (handling windows)
            from urllib.parse import urlparse, unquote
            parsed = urlparse(uri)
            path = unquote(parsed.path)
            # Window fix: /C:/User -> C:/User
            if os.name == 'nt' and path.startswith('/') and ':' in path:
                path = path[1:]
                
            start_line = loc.get("range", {}).get("start", {}).get("line", 0) + 1
            output.append(f"{path}:{start_line}")
            
        return "\n".join(output)

    except Exception as e:
        return f"<error>Failed to get definition: {e}</error>"
