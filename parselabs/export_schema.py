"""Shared export schema definitions for final CSV/Excel outputs."""

COLUMN_SCHEMA = {
    # Core identification
    "date": {"dtype": "datetime64[ns]", "excel_width": 13},
    # Extracted values (standardized)
    "lab_name": {"dtype": "str", "excel_width": 35},
    "value": {"dtype": "float64", "excel_width": 12},
    "lab_unit": {"dtype": "str", "excel_width": 15},
    # Source identification
    "source_file": {"dtype": "str", "excel_width": 25},
    "page_number": {"dtype": "Int64", "excel_width": 8},
    # Reference ranges from PDF
    "reference_min": {"dtype": "float64", "excel_width": 12},
    "reference_max": {"dtype": "float64", "excel_width": 12},
    # Raw values (for audit)
    "raw_lab_name": {"dtype": "str", "excel_width": 35},
    "raw_value": {"dtype": "str", "excel_width": 12},
    "raw_unit": {"dtype": "str", "excel_width": 15},
    # Review flags (from validation)
    "review_needed": {"dtype": "boolean", "excel_width": 12},
    "review_reason": {"dtype": "str", "excel_width": 30},
    # Limit indicators (for values like <0.05 or >738)
    "is_below_limit": {"dtype": "boolean", "excel_width": 12},
    "is_above_limit": {"dtype": "boolean", "excel_width": 12},
    # Internal (hidden in Excel)
    "lab_type": {"dtype": "str", "excel_width": 10, "excel_hidden": True},
    "result_index": {
        "dtype": "Int64",
        "excel_width": 10,
        "excel_hidden": True,
    },
}


COLUMN_ORDER = [
    "date",
    "lab_name",
    "value",
    "lab_unit",
    "source_file",
    "page_number",
    "reference_min",
    "reference_max",
    "raw_lab_name",
    "raw_value",
    "raw_unit",
    "review_needed",
    "review_reason",
    "is_below_limit",
    "is_above_limit",
    "lab_type",
    "result_index",
]


def get_column_lists(schema: dict):
    """Extract ordered export metadata from the schema definition."""

    export_cols = [k for k in COLUMN_ORDER if k in schema]
    hidden_cols = [col for col, props in schema.items() if props.get("excel_hidden")]
    widths = {col: props["excel_width"] for col, props in schema.items() if "excel_width" in props}
    dtypes = {col: props["dtype"] for col, props in schema.items() if "dtype" in props}
    return export_cols, hidden_cols, widths, dtypes
