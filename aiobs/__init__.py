"""Minimal LLM observability SDK.

Usage (global singleton):

    from aiobs import observer
    observer.observe()  # enable instrumentation
    # ... make LLM calls ...
    observer.end()      # end current session
    observer.flush()    # write a single JSON file to disk

Supported providers:
    - OpenAI (chat.completions)
    - Google Gemini (models.generate_content)

Function tracing with @observe decorator:

    from aiobs import observe

    @observe
    def my_function():
        ...

    @observe(name="custom_name")
    async def my_async_function():
        ...

Export to cloud storage or custom destinations:

    from aiobs import observer
    from aiobs.exporters import GCSExporter

    exporter = GCSExporter(
        bucket="my-observability-bucket",
        prefix="traces/",
        project="my-gcp-project",
    )
    observer.observe()
    # ... your agent code ...
    observer.end()
    observer.flush(exporter=exporter)
"""

from .collector import Collector
from .observe import observe
from .providers.base import BaseProvider

# Import models for type annotations and JSON parsing
from .models import (
    Session,
    SessionMeta,
    Event,
    FunctionEvent,
    ObservedEvent,
    ObservedFunctionEvent,
    ObservabilityExport,
    Callsite,
)

# Import exporter base classes (cloud-specific exporters use lazy imports)
from .exporters.base import BaseExporter, ExportResult, ExportError

# Import classifier base classes and models
from .classifier import (
    BaseClassifier,
    ClassificationConfig,
    ClassificationInput,
    ClassificationResult,
    ClassificationVerdict,
    OpenAIClassifier,
)

# Global collector singleton, intentionally simple API
observer = Collector()

# Public API exports - only expose supported interfaces
__all__ = [
    # Primary API
    "observer",
    "observe",
    # For custom provider development
    "Collector",
    "BaseProvider",
    # Exporter base classes
    "BaseExporter",
    "ExportResult",
    "ExportError",
    # Classifier base classes and models
    "BaseClassifier",
    "ClassificationConfig",
    "ClassificationInput",
    "ClassificationResult",
    "ClassificationVerdict",
    "OpenAIClassifier",
    # Data models (for parsing/validation)
    "Session",
    "SessionMeta",
    "Event",
    "FunctionEvent",
    "ObservedEvent",
    "ObservedFunctionEvent",
    "ObservabilityExport",
    "Callsite",
]
