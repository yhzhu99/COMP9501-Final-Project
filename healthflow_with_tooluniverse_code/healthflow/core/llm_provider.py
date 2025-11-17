import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from loguru import logger

from .config import LLMProviderConfig
from .token_tracker import TokenTracker, TokenUsage

class LLMMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    role: str
    content: str

class LLMResponse(BaseModel):
    """Represents a response from the LLM."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMProvider:
    """
    A wrapper around an OpenAI-compatible LLM client.
    Handles API calls with automatic retries for transient errors.
    """
    def __init__(self, config: LLMProviderConfig, token_tracker: Optional[TokenTracker] = None):
        """Initializes the provider with a specific LLM's configuration."""
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=httpx.AsyncClient(timeout=config.timeout),
        )
        self.model_name = config.model_name
        self.token_tracker = token_tracker
        logger.info(f"LLMProvider initialized for model: {self.model_name} at {config.base_url}")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        agent: str = "unknown",
    ) -> LLMResponse:
        """
        Generates a chat completion from the LLM.

        Args:
            messages: A list of LLMMessage objects representing the conversation history.
            temperature: The creativity of the response.
            max_tokens: The maximum number of tokens to generate.
            json_mode: Whether to enable JSON mode for the response.
            agent: The name of the agent making this call (for tracking purposes).

        Returns:
            An LLMResponse object with the generated content and token usage information.
        """
        logger.debug(f"Generating LLM response with model {self.model_name}. JSON mode: {json_mode}")
        try:
            # Set response format based on json_mode flag
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}

            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            content = completion.choices[0].message.content or ""
            
            # Extract token usage information
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            logger.info(f"LLM call completed for {agent}: {total_tokens} tokens (P:{prompt_tokens} C:{completion_tokens}) using {self.model_name}")
            
            # Create response with token information
            response = LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            # Record token usage if tracker is available
            if self.token_tracker:
                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=self.model_name,
                    agent=agent
                )
                self.token_tracker.record_usage(token_usage)
            
            return response
        except RetryError as e:
            logger.error(f"LLM API call failed after multiple retries: {e}")
            raise # Re-raise the exception after logging
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM API call: {e}")
            # This will trigger a retry if tenacity is configured for it
            raise

def create_llm_provider(config: LLMProviderConfig, token_tracker: Optional[TokenTracker] = None) -> LLMProvider:
    """Factory function to create the LLM provider from its specific config."""
    return LLMProvider(config, token_tracker)