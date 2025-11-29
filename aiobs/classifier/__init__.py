"""Classifier module for aiobs.

Provides classifiers to evaluate model response quality based on
system prompts, user inputs, and model outputs.

Example usage:
    from aiobs.classifier import OpenAIClassifier, ClassificationVerdict
    
    classifier = OpenAIClassifier(api_key="sk-...")
    result = classifier.classify(
        user_input="What is 2+2?",
        model_output="2+2 equals 4.",
        system_prompt="You are a math tutor."
    )
    
    if result.verdict == ClassificationVerdict.GOOD:
        print(f"Good response! Confidence: {result.confidence}")
    else:
        print(f"Issues found: {result.categories}")
"""

from .base import BaseClassifier
from .models import (
    ClassificationConfig,
    ClassificationInput,
    ClassificationResult,
    ClassificationVerdict,
)
from .openai import OpenAIClassifier

__all__ = [
    # Base class for custom classifiers
    "BaseClassifier",
    # Models
    "ClassificationConfig",
    "ClassificationInput",
    "ClassificationResult",
    "ClassificationVerdict",
    # Implementations
    "OpenAIClassifier",
]

