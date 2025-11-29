Classifier
==========

The classifier module provides tools to evaluate model response quality. Classifiers analyze
system prompts, user inputs, and model outputs to determine if responses are good, bad, or uncertain.

Overview
--------

Classifiers are useful for:

- **Quality assurance**: Automatically evaluate LLM outputs for correctness
- **Hallucination detection**: Identify factually incorrect responses
- **Relevance checking**: Detect off-topic or unhelpful answers
- **Batch evaluation**: Evaluate large datasets of model outputs

Quick Start
-----------

.. code-block:: python

   from aiobs.classifier import OpenAIClassifier, ClassificationVerdict

   classifier = OpenAIClassifier(api_key="sk-...")  # or uses OPENAI_API_KEY env var

   result = classifier.classify(
       system_prompt="You are a helpful math tutor.",
       user_input="What is 2 + 2?",
       model_output="2 + 2 equals 4."
   )

   if result.verdict == ClassificationVerdict.GOOD:
       print(f"Good response! Confidence: {result.confidence}")
   else:
       print(f"Issues: {result.categories}")
       print(f"Reasoning: {result.reasoning}")

Classification Models
---------------------

ClassificationInput
^^^^^^^^^^^^^^^^^^^

Input model containing the data to be classified:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``system_prompt``
     - ``Optional[str]``
     - The system prompt provided to the model
   * - ``user_input``
     - ``str``
     - The user's input/query to the model (required)
   * - ``model_output``
     - ``str``
     - The model's generated response (required)
   * - ``context``
     - ``Optional[Dict]``
     - Additional context for classification

ClassificationResult
^^^^^^^^^^^^^^^^^^^^

Result model returned by classifiers:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - ``verdict``
     - ``ClassificationVerdict``
     - The classification: ``GOOD``, ``BAD``, or ``UNCERTAIN``
   * - ``confidence``
     - ``float``
     - Confidence score between 0 and 1
   * - ``reasoning``
     - ``Optional[str]``
     - Explanation for the classification decision
   * - ``categories``
     - ``Optional[List[str]]``
     - Specific issues identified (e.g., "hallucination", "off-topic")
   * - ``raw_response``
     - ``Optional[Any]``
     - Raw response from the classification model
   * - ``metadata``
     - ``Optional[Dict]``
     - Additional metadata (model used, token usage, etc.)

ClassificationVerdict
^^^^^^^^^^^^^^^^^^^^^

Enum representing the classification outcome:

- ``ClassificationVerdict.GOOD`` - Response is correct, relevant, and helpful
- ``ClassificationVerdict.BAD`` - Response has issues (incorrect, off-topic, harmful)
- ``ClassificationVerdict.UNCERTAIN`` - Quality cannot be determined definitively

ClassificationConfig
^^^^^^^^^^^^^^^^^^^^

Configuration for classifier behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Field
     - Type
     - Default
     - Description
   * - ``model``
     - ``str``
     - ``"gpt-4o-mini"``
     - Model to use for classification
   * - ``temperature``
     - ``float``
     - ``0.0``
     - Temperature for classification model
   * - ``max_tokens``
     - ``int``
     - ``1024``
     - Maximum tokens for classification response
   * - ``classification_prompt``
     - ``Optional[str]``
     - ``None``
     - Custom prompt template for classification
   * - ``confidence_threshold``
     - ``float``
     - ``0.7``
     - Threshold for confident verdicts

OpenAI Classifier
-----------------

The ``OpenAIClassifier`` uses OpenAI's models to evaluate response quality.

Initialization
^^^^^^^^^^^^^^

.. code-block:: python

   from aiobs.classifier import OpenAIClassifier, ClassificationConfig

   # Using API key from environment (OPENAI_API_KEY)
   classifier = OpenAIClassifier()

   # With explicit API key
   classifier = OpenAIClassifier(api_key="sk-...")

   # With custom configuration
   config = ClassificationConfig(
       model="gpt-4o",
       temperature=0.0,
       max_tokens=2048,
   )
   classifier = OpenAIClassifier(config=config)

   # With pre-configured client
   from openai import OpenAI
   client = OpenAI(api_key="sk-...")
   classifier = OpenAIClassifier(client=client)

Synchronous Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   result = classifier.classify(
       user_input="What is the capital of France?",
       model_output="The capital of France is Paris.",
       system_prompt="You are a helpful geography assistant."
   )

   print(f"Verdict: {result.verdict.value}")  # "good"
   print(f"Confidence: {result.confidence}")   # 0.95
   print(f"Reasoning: {result.reasoning}")

Asynchronous Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio

   async def evaluate():
       result = await classifier.classify_async(
           user_input="What is 2+2?",
           model_output="2+2 equals 5."  # Wrong answer
       )
       return result

   result = asyncio.run(evaluate())
   print(f"Verdict: {result.verdict.value}")  # "bad"
   print(f"Categories: {result.categories}")  # ["incorrect_answer"]

Batch Classification
^^^^^^^^^^^^^^^^^^^^

Classify multiple responses efficiently:

.. code-block:: python

   from aiobs.classifier import ClassificationInput

   inputs = [
       ClassificationInput(
           system_prompt="You are a math tutor.",
           user_input="What is 10 * 5?",
           model_output="10 * 5 = 50."
       ),
       ClassificationInput(
           user_input="Tell me a joke.",
           model_output="Why did the chicken cross the road? To get to the other side!"
       ),
       ClassificationInput(
           system_prompt="You are a cooking assistant.",
           user_input="How do I make pasta?",
           model_output="The weather is sunny today."  # Off-topic
       ),
   ]

   # Synchronous batch (sequential)
   results = classifier.classify_batch(inputs)

   # Asynchronous batch (parallel) - recommended for large batches
   results = await classifier.classify_batch_async(inputs)

   for inp, result in zip(inputs, results):
       print(f"Q: {inp.user_input}")
       print(f"Verdict: {result.verdict.value}")
       print()

Custom Classifiers
------------------

Create custom classifiers by extending ``BaseClassifier``:

.. code-block:: python

   from aiobs.classifier import (
       BaseClassifier,
       ClassificationInput,
       ClassificationResult,
       ClassificationVerdict,
       ClassificationConfig,
   )
   from typing import Any, List, Optional

   class MyCustomClassifier(BaseClassifier):
       """Custom classifier using a different backend."""

       name = "custom"

       def __init__(self, config: Optional[ClassificationConfig] = None):
           super().__init__(config)
           # Initialize your backend here

       @classmethod
       def is_available(cls) -> bool:
           # Check if dependencies are available
           return True

       def classify(
           self,
           user_input: str,
           model_output: str,
           system_prompt: Optional[str] = None,
           **kwargs: Any,
       ) -> ClassificationResult:
           # Your classification logic here
           # Example: simple keyword-based classification
           if "error" in model_output.lower():
               return ClassificationResult(
                   verdict=ClassificationVerdict.BAD,
                   confidence=0.8,
                   reasoning="Response contains error indication",
                   categories=["error_response"],
               )
           return ClassificationResult(
               verdict=ClassificationVerdict.GOOD,
               confidence=0.7,
               reasoning="Response appears valid",
           )

       async def classify_async(
           self,
           user_input: str,
           model_output: str,
           system_prompt: Optional[str] = None,
           **kwargs: Any,
       ) -> ClassificationResult:
           # For simple cases, just call sync method
           return self.classify(user_input, model_output, system_prompt, **kwargs)

       def classify_batch(
           self,
           inputs: List[ClassificationInput],
           **kwargs: Any,
       ) -> List[ClassificationResult]:
           return [
               self.classify(
                   inp.user_input, inp.model_output, inp.system_prompt, **kwargs
               )
               for inp in inputs
           ]

       async def classify_batch_async(
           self,
           inputs: List[ClassificationInput],
           **kwargs: Any,
       ) -> List[ClassificationResult]:
           return self.classify_batch(inputs, **kwargs)

Examples
--------

The repository includes classifier examples at ``example/classifier/``:

- ``main.py`` - Synchronous classification examples
- ``async_example.py`` - Asynchronous batch classification

Run the examples::

   cd example/classifier
   python main.py
   python async_example.py

Evaluation Criteria
-------------------

The default classification prompt evaluates responses based on:

**Good responses:**

- Directly address the user's question or request
- Are factually accurate (no hallucinations)
- Are coherent and well-structured
- Follow system prompt guidelines (if provided)
- Are appropriate and helpful

**Bad responses:**

- Do not address the user's question
- Contain factual errors or hallucinations
- Are incoherent or poorly structured
- Violate system prompt guidelines
- Are inappropriate, harmful, or unhelpful

**Uncertain responses:**

- Quality cannot be determined definitively
- Response partially meets criteria
- More context would be needed to evaluate properly

Custom Classification Prompts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide a custom prompt template via ``ClassificationConfig``:

.. code-block:: python

   custom_prompt = """
   Evaluate if the AI response is helpful and accurate.

   User Question: {{ user_input }}
   AI Response: {{ model_output }}

   Respond with JSON: {"verdict": "good" or "bad", "confidence": 0.0-1.0, "reasoning": "..."}
   """

   config = ClassificationConfig(classification_prompt=custom_prompt)
   classifier = OpenAIClassifier(config=config)

