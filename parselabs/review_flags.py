"""Shared helpers for dataframe-backed review flags."""

from __future__ import annotations

import pandas as pd


def ensure_review_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure review flag columns exist with the legacy defaults."""

    # Add the boolean review marker when callers pass raw extracted rows.
    if "review_needed" not in df.columns:
        df["review_needed"] = False

    # Add the semicolon-formatted reason text when callers pass raw extracted rows.
    if "review_reason" not in df.columns:
        df["review_reason"] = ""

    return df


def append_review_reason_code(
    df: pd.DataFrame,
    mask: pd.Series,
    reason_code: str,
) -> pd.DataFrame:
    """Mark matching rows for review and append a reason code once."""

    # Guard: Empty frames or empty masks have nothing to update.
    if df.empty or not mask.any():
        return df

    # Prepare the shared review columns before setting any row flags.
    df = ensure_review_flag_columns(df)

    # Set the row-level review flag and append the legacy semicolon-delimited reason text.
    df.loc[mask, "review_needed"] = True
    current_reasons = df.loc[mask, "review_reason"].fillna("").astype(str)
    df.loc[mask, "review_reason"] = current_reasons.apply(
        lambda value: f"{value}{reason_code}; " if reason_code not in value else value
    )
    return df
