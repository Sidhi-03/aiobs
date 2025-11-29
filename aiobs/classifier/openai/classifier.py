"""OpenAI-based classifier implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional, Union

from ..base import BaseClassifier
from ..models import (
    ClassificationConfig,
    ClassificationInput,
    ClassificationResult,
    ClassificationVerdict,
)


class OpenAIClassifier(BaseClassifier):
    """Classifier using OpenAI's models to evaluate response quality.
    
    Uses OpenAI's chat completion API to analyze model outputs and
    determine if they are good, bad, or uncertain.
    
    Example:
        from aiobs.classifier import OpenAIClassifier
        
        classifier = OpenAIClassifier(api_key="sk-...")
        result = classifier.classify(
            user_input="What is the capital of France?",
            model_output="The capital of France is Paris.",
            system_prompt="You are a helpful geography assistant."
        )
        
        if result.verdict == ClassificationVerdict.GOOD:
            print("Response is good!")
        else:
            print(f"Issues: {result.categories}")
    """
    
    name = "openai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ClassificationConfig] = None,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
    ) -> None:
        """Initialize the OpenAI classifier.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            config: Classifier configuration. Defaults to ClassificationConfig().
            client: Optional pre-configured OpenAI client (sync).
            async_client: Optional pre-configured AsyncOpenAI client.
        """
        super().__init__(config)
        self._api_key = api_key
        self._client = client
        self._async_client = async_client
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI library is available."""
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False
    
    def _get_client(self) -> Any:
        """Get or create the synchronous OpenAI client."""
        if self._client is not None:
            return self._client
        
        from openai import OpenAI
        
        if self._api_key:
            self._client = OpenAI(api_key=self._api_key)
        else:
            self._client = OpenAI()  # Uses OPENAI_API_KEY env var
        
        return self._client
    
    def _get_async_client(self) -> Any:
        """Get or create the asynchronous OpenAI client."""
        if self._async_client is not None:
            return self._async_client
        
        from openai import AsyncOpenAI
        
        if self._api_key:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        else:
            self._async_client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var
        
        return self._async_client
    
    def _parse_response(self, content: str) -> ClassificationResult:
        """Parse the model response into a ClassificationResult.
        
        Args:
            content: Raw response content from the model.
        
        Returns:
            Parsed ClassificationResult.
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where model might wrap JSON in markdown code blocks
            clean_content = content.strip()
            if clean_content.startswith("```"):
                # Remove markdown code block
                lines = clean_content.split("\n")
                clean_content = "\n".join(lines[1:-1])
            
            data = json.loads(clean_content)
            
            verdict = self._parse_verdict(data.get("verdict", "uncertain"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning")
            categories = data.get("categories")
            
            return ClassificationResult(
                verdict=verdict,
                confidence=min(max(confidence, 0.0), 1.0),  # Clamp to [0, 1]
                reasoning=reasoning,
                categories=categories if categories else None,
                raw_response=content,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # If parsing fails, return uncertain with the raw response
            return ClassificationResult(
                verdict=ClassificationVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Failed to parse classification response: {str(e)}",
                raw_response=content,
                metadata={"parse_error": str(e)},
            )
    
    def classify(
        self,
        user_input: str,
        model_output: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """Classify a model response synchronously using OpenAI.
        
        Args:
            user_input: The user's input/query to the model.
            model_output: The model's generated response.
            system_prompt: Optional system prompt provided to the model.
            **kwargs: Additional arguments (passed to context).
        
        Returns:
            ClassificationResult with verdict, confidence, and reasoning.
        """
        classification_input = self._create_input(
            user_input=user_input,
            model_output=model_output,
            system_prompt=system_prompt,
            **kwargs,
        )
        
        prompt = self._format_prompt(classification_input)
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI response evaluator. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content or ""
            result = self._parse_response(content)
            result.metadata = result.metadata or {}
            result.metadata["model"] = self.config.model
            result.metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            return result
            
        except Exception as e:
            return ClassificationResult(
                verdict=ClassificationVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
    
    async def classify_async(
        self,
        user_input: str,
        model_output: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """Classify a model response asynchronously using OpenAI.
        
        Args:
            user_input: The user's input/query to the model.
            model_output: The model's generated response.
            system_prompt: Optional system prompt provided to the model.
            **kwargs: Additional arguments (passed to context).
        
        Returns:
            ClassificationResult with verdict, confidence, and reasoning.
        """
        classification_input = self._create_input(
            user_input=user_input,
            model_output=model_output,
            system_prompt=system_prompt,
            **kwargs,
        )
        
        prompt = self._format_prompt(classification_input)
        client = self._get_async_client()
        
        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI response evaluator. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content or ""
            result = self._parse_response(content)
            result.metadata = result.metadata or {}
            result.metadata["model"] = self.config.model
            result.metadata["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            return result
            
        except Exception as e:
            return ClassificationResult(
                verdict=ClassificationVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
    
    def classify_batch(
        self,
        inputs: List[ClassificationInput],
        **kwargs: Any,
    ) -> List[ClassificationResult]:
        """Classify multiple model responses in batch (sequential).
        
        Note: This runs classifications sequentially. For true parallel
        execution, use classify_batch_async.
        
        Args:
            inputs: List of ClassificationInput objects to classify.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            List of ClassificationResult objects, one per input.
        """
        results = []
        for inp in inputs:
            result = self.classify(
                user_input=inp.user_input,
                model_output=inp.model_output,
                system_prompt=inp.system_prompt,
                context=inp.context,
                **kwargs,
            )
            results.append(result)
        return results
    
    async def classify_batch_async(
        self,
        inputs: List[ClassificationInput],
        **kwargs: Any,
    ) -> List[ClassificationResult]:
        """Classify multiple model responses asynchronously in parallel.
        
        Uses asyncio.gather for concurrent classification requests.
        
        Args:
            inputs: List of ClassificationInput objects to classify.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            List of ClassificationResult objects, one per input.
        """
        tasks = [
            self.classify_async(
                user_input=inp.user_input,
                model_output=inp.model_output,
                system_prompt=inp.system_prompt,
                context=inp.context,
                **kwargs,
            )
            for inp in inputs
        ]
        return await asyncio.gather(*tasks)

