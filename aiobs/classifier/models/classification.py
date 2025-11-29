"""Pydantic models for classification inputs and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassificationVerdict(str, Enum):
    """Verdict for classification result."""
    
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"


class ClassificationInput(BaseModel):
    """Input model for classification.
    
    Contains the system prompt, user input, and model output
    that will be evaluated by the classifier.
    """
    
    system_prompt: Optional[str] = Field(
        default=None,
        description="The system prompt provided to the model"
    )
    user_input: str = Field(
        description="The user's input/query to the model"
    )
    model_output: str = Field(
        description="The model's generated response"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for classification (e.g., conversation history)"
    )


class ClassificationResult(BaseModel):
    """Result model for classification.
    
    Contains the verdict (good/bad/uncertain), confidence score,
    reasoning, and any additional metadata.
    """
    
    verdict: ClassificationVerdict = Field(
        description="The classification verdict: good, bad, or uncertain"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation for the classification decision"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Specific categories/issues identified (e.g., 'hallucination', 'off-topic')"
    )
    raw_response: Optional[Any] = Field(
        default=None,
        description="Raw response from the classification model"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata from the classifier"
    )


class ClassificationConfig(BaseModel):
    """Configuration for classifier behavior."""
    
    model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for classification"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for classification model"
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for classification response"
    )
    classification_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt template for classification"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold above which verdict is considered confident"
    )

