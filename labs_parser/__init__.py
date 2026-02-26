"""Labs Parser - Medical lab report extraction and processing."""

from labs_parser.config import (
    UNKNOWN_VALUE,
    Demographics,
    ExtractionConfig,
    LabSpecsConfig,
    ProfileConfig,
)
from labs_parser.exceptions import ConfigurationError, PipelineError
from labs_parser.extraction import (
    HealthLabReport,
    LabResult,
    extract_labs_from_page_image,
    extract_labs_from_text,
)
from labs_parser.normalization import (
    apply_dtype_conversions,
    apply_normalizations,
    deduplicate_results,
)
from labs_parser.standardization import (
    standardize_lab_names,
    standardize_lab_units,
)
from labs_parser.utils import (
    ensure_columns,
    load_dotenv_with_env,
    parse_llm_json_response,
    preprocess_page_image,
    setup_logging,
)
from labs_parser.validation import ValueValidator

__version__ = "0.1.0"

__all__ = [
    # Exceptions
    "ConfigurationError",
    "PipelineError",
    # Config
    "ExtractionConfig",
    "ProfileConfig",
    "LabSpecsConfig",
    "Demographics",
    "UNKNOWN_VALUE",
    # Extraction
    "LabResult",
    "HealthLabReport",
    "extract_labs_from_page_image",
    "extract_labs_from_text",
    # Validation
    "ValueValidator",
    # Standardization
    "standardize_lab_names",
    "standardize_lab_units",
    # Normalization
    "apply_normalizations",
    "deduplicate_results",
    "apply_dtype_conversions",
    # Utils
    "load_dotenv_with_env",
    "preprocess_page_image",
    "setup_logging",
    "ensure_columns",
    "parse_llm_json_response",
]
