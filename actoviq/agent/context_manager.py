"""
Klynx Agent - Token 

 prompt , token .
 token  usage.
"""

import re
from typing import List
from langchain_core.messages import BaseMessage


class TokenCounter:
    """Token ()"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        token
        :0.6 token/
        :0.3 token/
        """
        if not text:
            return 0
        
        # 
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        
        #  0.6 token/, 0.3 token/
        tokens = chinese_chars * 0.6 + other_chars * 0.3
        return int(tokens)
    
    @staticmethod
    def count_message_tokens(messages: List[BaseMessage]) -> int:
        """token"""
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            total += TokenCounter.estimate_tokens(content)
        return total
