"""Labs Parser - Medical lab report extraction and processing."""

from labs_parser.config import (
    ExtractionConfig,
    ProfileConfig,
    LabSpecsConfig,
    Demographics,
    UNKNOWN_VALUE,
)
from labs_parser.extraction import (
    LabResult,
    HealthLabReport,
    extract_labs_from_page_image,
    extract_labs_from_text,
    self_consistency,
)
from labs_parser.validation import ValueValidator
from labs_parser.standardization import standardize_lab_names, standardize_lab_units
from labs_parser.normalization import (
    apply_normalizations,
    deduplicate_results,
    apply_dtype_conversions,
)
from labs_parser.utils import (
    load_dotenv_with_env,
    preprocess_page_image,
    setup_logging,
    ensure_columns,
    parse_llm_json_response,
)

__version__ = "0.1.0"

__all__ = [
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
    "self_consistency",
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
