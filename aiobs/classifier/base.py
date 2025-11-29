"""Base classifier interface for aiobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .models import (
    ClassificationConfig,
    ClassificationInput,
    ClassificationResult,
    ClassificationVerdict,
)


class BaseClassifier(ABC):
    """Abstract base class for response classifiers.
    
    Classifiers evaluate model outputs against user inputs and system prompts
    to determine if the response quality is good, bad, or uncertain.
    
    Subclasses must implement:
        - classify(): Synchronous classification
        - classify_async(): Asynchronous classification
        - classify_batch(): Batch classification for multiple inputs
    
    Example usage:
        from aiobs.classifier import OpenAIClassifier
        
        classifier = OpenAIClassifier(api_key="...")
        result = classifier.classify(
            system_prompt="You are a helpful assistant.",
            user_input="What is 2+2?",
            model_output="2+2 equals 4."
        )
        print(result.verdict)  # ClassificationVerdict.GOOD
    """
    
    name: str = "base"
    
    def __init__(self, config: Optional[ClassificationConfig] = None) -> None:
        """Initialize the classifier with optional configuration.
        
        Args:
            config: Configuration for classifier behavior. If None, uses defaults.
        """
        self.config = config or ClassificationConfig()
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this classifier can be used (dependencies present).
        
        Returns:
            True if all required dependencies are available.
        """
        return True
    
    @abstractmethod
    def classify(
        self,
        user_input: str,
        model_output: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """Classify a model response synchronously.
        
        Args:
            user_input: The user's input/query to the model.
            model_output: The model's generated response.
            system_prompt: Optional system prompt provided to the model.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            ClassificationResult with verdict, confidence, and reasoning.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def classify_async(
        self,
        user_input: str,
        model_output: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationResult:
        """Classify a model response asynchronously.
        
        Args:
            user_input: The user's input/query to the model.
            model_output: The model's generated response.
            system_prompt: Optional system prompt provided to the model.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            ClassificationResult with verdict, confidence, and reasoning.
        """
        raise NotImplementedError
    
    @abstractmethod
    def classify_batch(
        self,
        inputs: List[ClassificationInput],
        **kwargs: Any,
    ) -> List[ClassificationResult]:
        """Classify multiple model responses in batch.
        
        Args:
            inputs: List of ClassificationInput objects to classify.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            List of ClassificationResult objects, one per input.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def classify_batch_async(
        self,
        inputs: List[ClassificationInput],
        **kwargs: Any,
    ) -> List[ClassificationResult]:
        """Classify multiple model responses asynchronously in batch.
        
        Args:
            inputs: List of ClassificationInput objects to classify.
            **kwargs: Additional arguments for the classifier.
        
        Returns:
            List of ClassificationResult objects, one per input.
        """
        raise NotImplementedError
    
    def _create_input(
        self,
        user_input: str,
        model_output: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ClassificationInput:
        """Helper to create ClassificationInput from individual arguments.
        
        Args:
            user_input: The user's input/query.
            model_output: The model's response.
            system_prompt: Optional system prompt.
            **kwargs: Additional context.
        
        Returns:
            ClassificationInput instance.
        """
        return ClassificationInput(
            system_prompt=system_prompt,
            user_input=user_input,
            model_output=model_output,
            context=kwargs.get("context"),
        )
    
    def _default_classification_prompt(self) -> str:
        """Get the default prompt template for classification.
        
        Returns:
            Default classification prompt template.
        """
        return """You are an expert AI response evaluator. Your task is to assess whether a model's response is good or bad based on the given context.

## Evaluation Criteria

A response is considered **GOOD** if it:
- Directly addresses the user's question or request
- Is factually accurate (no hallucinations)
- Is coherent and well-structured
- Follows the system prompt guidelines (if provided)
- Is appropriate and helpful

A response is considered **BAD** if it:
- Does not address the user's question
- Contains factual errors or hallucinations
- Is incoherent or poorly structured
- Violates system prompt guidelines
- Is inappropriate, harmful, or unhelpful

A response is **UNCERTAIN** if:
- The quality cannot be determined definitively
- The response partially meets criteria
- More context would be needed to evaluate properly

## Input

{%- if system_prompt %}
### System Prompt
{{ system_prompt }}
{% endif %}

### User Input
{{ user_input }}

### Model Output
{{ model_output }}

## Your Task

Evaluate the model output and provide:
1. verdict: "good", "bad", or "uncertain"
2. confidence: A score from 0.0 to 1.0 indicating your confidence
3. reasoning: A brief explanation of your evaluation
4. categories: A list of specific issues if verdict is "bad" (e.g., ["hallucination", "off-topic"])

Respond ONLY with valid JSON in this exact format:
{
    "verdict": "good" | "bad" | "uncertain",
    "confidence": <float between 0 and 1>,
    "reasoning": "<your explanation>",
    "categories": ["<issue1>", "<issue2>"] or null
}"""

    def _format_prompt(
        self,
        classification_input: ClassificationInput,
        prompt_template: Optional[str] = None,
    ) -> str:
        """Format the classification prompt with input data.
        
        Uses Jinja2-style templating if available, otherwise simple replacement.
        
        Args:
            classification_input: The input to classify.
            prompt_template: Optional custom prompt template.
        
        Returns:
            Formatted prompt string.
        """
        template = prompt_template or self.config.classification_prompt or self._default_classification_prompt()
        
        # Simple string replacement (works without Jinja2)
        prompt = template
        
        if classification_input.system_prompt:
            prompt = prompt.replace("{{ system_prompt }}", classification_input.system_prompt)
            prompt = prompt.replace("{%- if system_prompt %}", "")
            prompt = prompt.replace("{% endif %}", "")
        else:
            # Remove the system prompt section if not provided
            import re
            prompt = re.sub(
                r"\{%- if system_prompt %\}.*?\{% endif %\}",
                "",
                prompt,
                flags=re.DOTALL
            )
        
        prompt = prompt.replace("{{ user_input }}", classification_input.user_input)
        prompt = prompt.replace("{{ model_output }}", classification_input.model_output)
        
        return prompt.strip()
    
    def _parse_verdict(self, verdict_str: str) -> ClassificationVerdict:
        """Parse a verdict string to ClassificationVerdict enum.
        
        Args:
            verdict_str: String verdict ("good", "bad", "uncertain").
        
        Returns:
            ClassificationVerdict enum value.
        """
        verdict_lower = verdict_str.lower().strip()
        if verdict_lower == "good":
            return ClassificationVerdict.GOOD
        elif verdict_lower == "bad":
            return ClassificationVerdict.BAD
        else:
            return ClassificationVerdict.UNCERTAIN

