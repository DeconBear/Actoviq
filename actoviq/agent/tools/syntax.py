import ast
import json
import os
import subprocess
from typing import Tuple

class SyntaxChecker:
    """
    
    : Python, JSON, JavaScript (via node)
    """
    
    @staticmethod
    def check_file(path: str) -> str:
        """
        
        
        Returns:
            
        """
        if not os.path.exists(path):
            return f"<error>: {path}</error>"
            
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.py':
            return SyntaxChecker._check_python(path)
        elif ext == '.json':
            return SyntaxChecker._check_json(path)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return SyntaxChecker._check_javascript(path)
        else:
            return f"<warning> {ext} </warning>"

    @staticmethod
    def _check_python(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content, filename=path)
            return f"<success>Python : {path}</success>"
        except SyntaxError as e:
            return f"<error>Python :\n: {path}\n: {e.lineno}\n: {e.offset}\n: {e.msg}\n: {e.text.strip() if e.text else ''}</error>"
        except Exception as e:
            return f"<error>: {str(e)}</error>"

    @staticmethod
    def _check_json(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            return f"<success>JSON : {path}</success>"
        except json.JSONDecodeError as e:
            return f"<error>JSON :\n: {path}\n: {e.lineno}\n: {e.colno}\n: {e.msg}</error>"
        except Exception as e:
            return f"<error>: {str(e)}</error>"

    @staticmethod
    def _check_javascript(path: str) -> str:
        #  node
        try:
            result = subprocess.run(
                ['node', '--check', path],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True # Windows  shell=True  node
            )
            
            if result.returncode == 0:
                return f"<success>JS/TS : {path}</success>"
            else:
                # 
                stderr = result.stderr.strip()
                #  stderr, stdout
                output = stderr if stderr else result.stdout.strip()
                return f"<error>JS/TS :\n{output}</error>"
                
        except FileNotFoundError:
            return "<warning> node , JS . Node.js.</warning>"
        except subprocess.TimeoutExpired:
            return "<error></error>"
        except Exception as e:
            return f"<error>: {str(e)}</error>"
