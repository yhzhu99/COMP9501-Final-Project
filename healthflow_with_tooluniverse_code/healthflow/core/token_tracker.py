import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class TokenUsage:
    """Represents token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    agent: str = ""

@dataclass 
class TokenSummary:
    """Summary of token usage across all calls."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    calls_by_agent: Dict[str, int] = field(default_factory=dict)
    tokens_by_agent: Dict[str, int] = field(default_factory=dict)

class TokenTracker:
    """
    Thread-safe token usage tracker for HealthFlow.
    Tracks token consumption across all LLM calls and provides detailed summaries.
    """
    
    def __init__(self):
        self._usage_history: list[TokenUsage] = []
        self._lock = threading.Lock()
        
    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage from a single LLM call."""
        with self._lock:
            self._usage_history.append(usage)
            logger.debug(f"Token usage recorded: {usage.total_tokens} tokens for {usage.agent} using {usage.model}")
    
    def get_summary(self) -> TokenSummary:
        """Get a summary of all token usage."""
        with self._lock:
            summary = TokenSummary()
            
            for usage in self._usage_history:
                summary.total_prompt_tokens += usage.prompt_tokens
                summary.total_completion_tokens += usage.completion_tokens
                summary.total_tokens += usage.total_tokens
                
                # Track by agent
                agent = usage.agent or "unknown"
                summary.calls_by_agent[agent] = summary.calls_by_agent.get(agent, 0) + 1
                summary.tokens_by_agent[agent] = summary.tokens_by_agent.get(agent, 0) + usage.total_tokens
            
            return summary
    
    def get_formatted_summary(self) -> str:
        """Get a formatted string summary of token usage."""
        summary = self.get_summary()
        
        lines = [
            "📊 Token Consumption Summary",
            "=" * 30,
            f"Total Prompt Tokens: {summary.total_prompt_tokens:,}",
            f"Total Completion Tokens: {summary.total_completion_tokens:,}",
            f"Total Tokens: {summary.total_tokens:,}",
            "",
            "Usage by Agent:",
        ]
        
        for agent, tokens in summary.tokens_by_agent.items():
            calls = summary.calls_by_agent.get(agent, 0)
            lines.append(f"  {agent}: {tokens:,} tokens ({calls} calls)")
        
        return "\n".join(lines)
    
    def get_log_summary(self) -> str:
        """Get a compact summary suitable for logging."""
        summary = self.get_summary()
        agent_details = ", ".join([
            f"{agent}:{tokens}t/{calls}c" 
            for agent, tokens in summary.tokens_by_agent.items() 
            for calls in [summary.calls_by_agent.get(agent, 0)]
        ])
        return f"Tokens: {summary.total_tokens:,} (P:{summary.total_prompt_tokens:,} C:{summary.total_completion_tokens:,}) Agents: {agent_details}"
    
    def reset(self) -> None:
        """Clear all recorded usage."""
        with self._lock:
            self._usage_history.clear()
            logger.debug("Token tracker reset")