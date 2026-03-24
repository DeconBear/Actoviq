"""

"""


def format_tool_output(tool_name: str, output: str) -> str:
    """
    
    
    Args:
        tool_name: 
        output: 
        
    Returns:
        
    """
    lines = [
        f"┌─ : {tool_name}",
        "│",
    ]
    
    for line in output.split('\n'):
        lines.append(f"│ {line}")
    
    lines.append("└─")
    
    return '\n'.join(lines)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    
    
    Args:
        text: 
        max_length: 
        suffix: 
        
    Returns:
        
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
