"""
Post-Extraction Verification for Lab Results

Multi-model verification pipeline to ensure extraction accuracy by:
1. Cross-model extraction (extract with different model, compare results)
2. Value-by-value verification (targeted prompts for each value)
3. Character-level reading (digit-by-digit verification for uncertain values)
4. Completeness check (detect missed results)
5. Disagreement arbitration (third model resolves conflicts)

All verification uses OpenRouter models - no local OCR dependencies.
"""

import json
import base64
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

logger = logging.getLogger(__name__)


# ========================================
# Model Configuration
# ========================================

# Vision-capable models by provider for cross-validation
# Updated January 2025 - using latest models from each provider
# Models are ordered by preference (best first) within each provider
#
# Sources:
# - https://openrouter.ai/anthropic
# - https://openrouter.ai/google
# - https://openrouter.ai/openai
# - https://openrouter.ai/qwen

VERIFICATION_MODELS = {
    # Anthropic Claude series
    # Claude Opus 4.5 is the frontier reasoning model
    "anthropic": [
        "anthropic/claude-opus-4.5",      # Frontier model, best quality ($5/M in, $25/M out)
        "anthropic/claude-sonnet-4",      # Good balance of speed/quality
        "anthropic/claude-haiku-4.5",     # Fast/cheap fallback
    ],

    # Google Gemini series
    # Gemini 3 Flash Preview is latest, 2.5 Flash is stable workhorse
    "google": [
        "google/gemini-3-flash-preview",  # Latest preview, 1M context ($0.50/M in, $3/M out)
        "google/gemini-2.5-flash",        # Stable workhorse ($0.30/M in, $2.50/M out)
        "google/gemini-2.5-pro",          # Higher quality alternative
    ],

    # OpenAI GPT series
    "openai": [
        "openai/gpt-5.2",                 # Latest model, 400K context ($1.75/M in, $14/M out)
        "openai/gpt-4.1",                 # 1M context, good fallback
        "openai/gpt-4.1-mini",            # Cost-effective
    ],

    # Qwen series
    # Strong models with excellent multilingual support and OCR
    "qwen": [
        "qwen/qwen3-max",                 # SOTA model, 256K context ($1.20/M in, $6/M out)
        "qwen/qwen3-vl-32b-instruct",     # Vision-specific model
        "qwen/qwen3-vl-8b-instruct",      # Faster/cheaper vision
    ],
}

# Provider priority for automatic model selection
# Order based on: vision accuracy, availability, cost-effectiveness
PROVIDER_PRIORITY = ["anthropic", "google", "openai", "qwen"]


def get_provider(model_id: str) -> str:
    """Extract provider name from model ID."""
    return model_id.split("/")[0].lower()


def get_verification_model(primary_model: str) -> str:
    """
    Select a verification model from a different provider than the primary model.

    Cross-provider verification catches provider-specific biases and errors.
    """
    primary_provider = get_provider(primary_model)

    # Find first available provider different from primary
    for provider in PROVIDER_PRIORITY:
        if provider != primary_provider and provider in VERIFICATION_MODELS:
            return VERIFICATION_MODELS[provider][0]

    # Fallback: use a different model from same provider
    if primary_provider in VERIFICATION_MODELS:
        models = VERIFICATION_MODELS[primary_provider]
        for model in models:
            if model != primary_model:
                return model

    # Last resort
    return "anthropic/claude-sonnet-4"


def get_arbitration_model(model1: str, model2: str) -> str:
    """Select a third model for arbitration, different from both input models."""
    used_providers = {get_provider(model1), get_provider(model2)}

    for provider, models in VERIFICATION_MODELS.items():
        if provider not in used_providers:
            return models[0]

    # If all providers used, pick the strongest available
    return "anthropic/claude-sonnet-4"


# ========================================
# Data Classes
# ========================================

@dataclass
class ValueVerification:
    """Verification result for a single lab result."""
    lab_name_raw: str
    value_raw: Optional[str]
    lab_unit_raw: Optional[str]

    # Verification status
    status: str = "pending"  # verified, mismatch, not_found, uncertain

    # Cross-model comparison
    cross_model_value: Optional[str] = None
    cross_model_agrees: Optional[bool] = None

    # Character-level verification
    character_verified: Optional[bool] = None
    character_reading: Optional[str] = None

    # Final determination
    verified_value: Optional[str] = None
    confidence: float = 1.0
    verification_method: str = ""
    notes: List[str] = field(default_factory=list)


@dataclass
class VerificationSummary:
    """Summary statistics for page verification."""
    total_results: int = 0
    verified_count: int = 0
    mismatch_count: int = 0
    corrected_count: int = 0
    uncertain_count: int = 0
    missed_count: int = 0
    avg_confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "total": self.total_results,
            "verified": self.verified_count,
            "verified_pct": (self.verified_count / self.total_results * 100) if self.total_results else 0,
            "mismatches": self.mismatch_count,
            "corrected": self.corrected_count,
            "uncertain": self.uncertain_count,
            "missed": self.missed_count,
            "avg_confidence": self.avg_confidence,
        }


# ========================================
# Verification Prompts
# ========================================

CROSS_EXTRACTION_SYSTEM = """You are a medical lab report data extractor focused on ACCURACY.

CRITICAL RULES:
1. Extract test names, values, and units EXACTLY as written - preserve all formatting
2. For value_raw: Extract the EXACT number or text shown (e.g., "5.2", "NEGATIVO", "142")
3. For lab_unit_raw: Extract the EXACT unit as shown (e.g., "mg/dL", "x10^9/L", "%")
4. Include ALL test results - don't skip any
5. For reference ranges: Copy exactly AND parse min/max values

Your extraction will be compared against another model's extraction to catch errors.
""".strip()


VALUE_VERIFICATION_PROMPT = """You are verifying extracted lab values against their source image.

I have extracted lab results from this image. For EACH result below, verify by looking at the ACTUAL IMAGE:

{results_table}

For EACH result, find it in the image and report:
1. Can you find this test name in the image? (yes/no)
2. What EXACT value do you see for this test? (read character-by-character)
3. Does the extracted value match what you see? (match/mismatch/uncertain)
4. What unit do you see? (exact text)

CRITICAL: Read each value DIGIT BY DIGIT. For "142.5", read as: one-four-two-point-five.

Respond with JSON array:
[
  {{
    "lab_name_raw": "original name from input",
    "found_in_image": true/false,
    "actual_value_seen": "exact value you read from image",
    "value_matches": "match" | "mismatch" | "uncertain",
    "actual_unit_seen": "exact unit from image",
    "unit_matches": "match" | "mismatch" | "uncertain",
    "confidence": 0.0-1.0,
    "location": "description of where in image"
  }}
]
""".strip()


CHARACTER_LEVEL_PROMPT = """You are performing character-by-character verification of a specific lab value.

Test: {test_name}
Extracted Value: {extracted_value}

TASK: Find this EXACT test in the image and read its value CHARACTER BY CHARACTER.

Instructions:
1. Locate the test "{test_name}" in the image
2. Find the numeric/text value associated with it
3. Read EACH CHARACTER separately: digits, decimal points, letters
4. Report what you see

Example for value "142.5":
- Character 1: "1"
- Character 2: "4"
- Character 3: "2"
- Character 4: "."
- Character 5: "5"
- Full value: "142.5"

Respond with JSON:
{{
  "test_found": true/false,
  "characters": ["1", "4", "2", ".", "5"],
  "full_value": "142.5",
  "matches_extracted": true/false,
  "confidence": 0.0-1.0,
  "notes": "any observations"
}}
""".strip()


COMPLETENESS_CHECK_PROMPT = """You are checking if any lab results were MISSED during extraction.

I extracted these lab tests from the image:
{extracted_names}

TASK: Look at the image and identify ANY lab tests that are NOT in the list above.

For each MISSED test, extract:
- lab_name_raw: Test name exactly as shown
- value_raw: Result value exactly as shown
- lab_unit_raw: Unit exactly as shown
- reference_range: Reference range if shown

Respond with JSON:
{{
  "all_tests_found": true/false,
  "missed_tests": [
    {{
      "lab_name_raw": "...",
      "value_raw": "...",
      "lab_unit_raw": "...",
      "reference_range": "..."
    }}
  ],
  "notes": "explanation"
}}
""".strip()


ARBITRATION_PROMPT = """You are resolving a disagreement between two AI extractions.

Test Name: {test_name}
Model A extracted value: {value_a}
Model B extracted value: {value_b}

TASK: Look at the ACTUAL IMAGE and determine which value is correct.

Instructions:
1. Find the test "{test_name}" in the image
2. Read the value VERY CAREFULLY, character by character
3. Determine which extraction (A or B) is correct, or if neither is correct

Respond with JSON:
{{
  "correct_source": "A" | "B" | "neither",
  "actual_value": "what you see in the image",
  "confidence": 0.0-1.0,
  "reasoning": "explanation of your determination"
}}
""".strip()


# ========================================
# Core Verification Functions
# ========================================

def encode_image(image_path: Path) -> str:
    """Encode image as base64 for API calls."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def call_vision_model(
    client: OpenAI,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    image_b64: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Optional[str]:
    """Make a vision model API call."""
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision model call failed ({model_id}): {e}")
        return None


def parse_json_response(text: str) -> Optional[Any]:
    """Parse JSON from model response, handling markdown fences."""
    if not text:
        return None

    # Remove markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split('\n')
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        text = '\n'.join(lines).strip()

    # Try to find JSON array or object
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    return None


# ========================================
# Stage 1: Cross-Model Extraction
# ========================================

def cross_model_extract(
    image_path: Path,
    client: OpenAI,
    verification_model: str,
) -> List[dict]:
    """
    Extract lab results using verification model for comparison.

    Returns list of extracted results from the verification model.
    """
    from extraction import EXTRACTION_USER_PROMPT, TOOLS, HealthLabReport

    image_b64 = encode_image(image_path)

    try:
        completion = client.chat.completions.create(
            model=verification_model,
            messages=[
                {"role": "system", "content": CROSS_EXTRACTION_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": EXTRACTION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            temperature=0.0,
            max_tokens=16384,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "extract_lab_results"}}
        )

        if not completion.choices[0].message.tool_calls:
            logger.warning(f"Cross-model extraction returned no tool call")
            return []

        tool_args = completion.choices[0].message.tool_calls[0].function.arguments
        result = json.loads(tool_args)
        return result.get("lab_results", [])

    except Exception as e:
        logger.error(f"Cross-model extraction failed: {e}")
        return []


def compare_extractions(
    primary_results: List[dict],
    verification_results: List[dict],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Compare primary and verification extractions.

    Returns:
        Tuple of (matching, disagreements, only_in_verification)
    """
    # Build lookup by normalized test name
    def normalize_name(name: str) -> str:
        if not name:
            return ""
        return re.sub(r'\s+', ' ', name.lower().strip())

    verification_map = {}
    for v in verification_results:
        key = normalize_name(v.get("lab_name_raw", ""))
        if key:
            verification_map[key] = v

    matching = []
    disagreements = []
    matched_keys = set()

    for primary in primary_results:
        primary_name = normalize_name(primary.get("lab_name_raw", ""))
        if not primary_name:
            continue

        if primary_name in verification_map:
            verification = verification_map[primary_name]
            matched_keys.add(primary_name)

            # Compare values
            p_val = str(primary.get("value_raw", "")).strip()
            v_val = str(verification.get("value_raw", "")).strip()

            # Normalize for comparison (handle decimal variations)
            def normalize_value(val: str) -> str:
                val = val.replace(",", ".")
                try:
                    num = float(val)
                    # Round to avoid floating point issues
                    return f"{num:.6f}".rstrip('0').rstrip('.')
                except ValueError:
                    return val.lower().strip()

            if normalize_value(p_val) == normalize_value(v_val):
                matching.append({
                    "primary": primary,
                    "verification": verification,
                    "status": "match"
                })
            else:
                disagreements.append({
                    "primary": primary,
                    "verification": verification,
                    "primary_value": p_val,
                    "verification_value": v_val,
                    "status": "disagreement"
                })
        else:
            # Not found in verification - could be extraction difference
            disagreements.append({
                "primary": primary,
                "verification": None,
                "primary_value": str(primary.get("value_raw", "")),
                "verification_value": None,
                "status": "not_in_verification"
            })

    # Find results only in verification (potentially missed by primary)
    only_in_verification = [
        v for key, v in verification_map.items()
        if key not in matched_keys
    ]

    return matching, disagreements, only_in_verification


# ========================================
# Stage 2: Value-by-Value Verification
# ========================================

def verify_values_batch(
    image_path: Path,
    results_to_verify: List[dict],
    client: OpenAI,
    model_id: str,
) -> List[dict]:
    """
    Verify multiple values in a single API call.

    More efficient than per-value calls for initial verification.
    """
    if not results_to_verify:
        return []

    image_b64 = encode_image(image_path)

    # Build results table for prompt
    table_lines = ["| Test Name | Extracted Value | Extracted Unit |", "|---|---|---|"]
    for r in results_to_verify:
        name = r.get("lab_name_raw", "")
        value = r.get("value_raw", "")
        unit = r.get("lab_unit_raw", "")
        table_lines.append(f"| {name} | {value} | {unit} |")

    results_table = "\n".join(table_lines)
    prompt = VALUE_VERIFICATION_PROMPT.format(results_table=results_table)

    response = call_vision_model(
        client, model_id,
        "You are a precise medical data verification expert.",
        prompt, image_b64,
        temperature=0.0
    )

    if not response:
        return []

    verifications = parse_json_response(response)
    if isinstance(verifications, dict) and "verifications" in verifications:
        verifications = verifications["verifications"]

    return verifications if isinstance(verifications, list) else []


# ========================================
# Stage 3: Character-Level Verification
# ========================================

def verify_value_character_level(
    image_path: Path,
    test_name: str,
    extracted_value: str,
    client: OpenAI,
    model_id: str,
) -> dict:
    """
    Character-by-character verification of a specific value.

    Used for high-priority disagreements.
    """
    image_b64 = encode_image(image_path)

    prompt = CHARACTER_LEVEL_PROMPT.format(
        test_name=test_name,
        extracted_value=extracted_value
    )

    response = call_vision_model(
        client, model_id,
        "You are performing precise character-by-character verification. Be extremely careful.",
        prompt, image_b64,
        temperature=0.0
    )

    if not response:
        return {"test_found": False, "confidence": 0.0}

    result = parse_json_response(response)
    return result if isinstance(result, dict) else {"test_found": False, "confidence": 0.0}


# ========================================
# Stage 4: Completeness Check
# ========================================

def check_completeness(
    image_path: Path,
    extracted_results: List[dict],
    client: OpenAI,
    model_id: str,
) -> List[dict]:
    """
    Check if any lab tests were missed during extraction.

    Returns list of missed tests.
    """
    image_b64 = encode_image(image_path)

    # Get extracted test names
    extracted_names = [r.get("lab_name_raw", "") for r in extracted_results if r.get("lab_name_raw")]
    names_list = "\n".join(f"- {name}" for name in extracted_names)

    prompt = COMPLETENESS_CHECK_PROMPT.format(extracted_names=names_list)

    response = call_vision_model(
        client, model_id,
        "You are checking for missed lab tests. Be thorough.",
        prompt, image_b64,
        temperature=0.0
    )

    if not response:
        return []

    result = parse_json_response(response)
    if isinstance(result, dict):
        return result.get("missed_tests", [])

    return []


# ========================================
# Stage 5: Disagreement Arbitration
# ========================================

def arbitrate_disagreement(
    image_path: Path,
    test_name: str,
    value_a: str,
    value_b: str,
    client: OpenAI,
    arbitration_model: str,
) -> dict:
    """
    Use a third model to resolve a disagreement.
    """
    image_b64 = encode_image(image_path)

    prompt = ARBITRATION_PROMPT.format(
        test_name=test_name,
        value_a=value_a,
        value_b=value_b
    )

    response = call_vision_model(
        client, arbitration_model,
        "You are an arbitrator resolving extraction disagreements. Be extremely precise.",
        prompt, image_b64,
        temperature=0.0
    )

    if not response:
        return {"correct_source": "uncertain", "confidence": 0.0}

    result = parse_json_response(response)
    return result if isinstance(result, dict) else {"correct_source": "uncertain", "confidence": 0.0}


# ========================================
# Main Verification Pipeline
# ========================================

class ExtractionVerifier:
    """
    Multi-model verification pipeline for extraction accuracy.

    Stages:
    1. Cross-model extraction (extract with different model)
    2. Compare and identify disagreements
    3. Value-by-value verification for disagreements
    4. Character-level verification for high-priority items
    5. Completeness check for missed results
    6. Arbitration for unresolved disagreements
    """

    def __init__(
        self,
        client: OpenAI,
        primary_model: str,
        verification_model: Optional[str] = None,
        arbitration_model: Optional[str] = None,
        enable_completeness_check: bool = True,
        enable_character_verification: bool = True,
        parallel_verification: bool = True,
    ):
        """
        Initialize verifier.

        Args:
            client: OpenAI client configured for OpenRouter
            primary_model: Model used for primary extraction
            verification_model: Model for cross-extraction (auto-selected if None)
            arbitration_model: Model for resolving disagreements (auto-selected if None)
            enable_completeness_check: Check for missed results
            enable_character_verification: Do character-level verification
            parallel_verification: Run verifications in parallel where possible
        """
        self.client = client
        self.primary_model = primary_model
        self.verification_model = verification_model or get_verification_model(primary_model)
        self.arbitration_model = arbitration_model or get_arbitration_model(
            primary_model, self.verification_model
        )
        self.enable_completeness_check = enable_completeness_check
        self.enable_character_verification = enable_character_verification
        self.parallel_verification = parallel_verification

        logger.info(f"Verifier initialized:")
        logger.info(f"  Primary model: {primary_model}")
        logger.info(f"  Verification model: {self.verification_model}")
        logger.info(f"  Arbitration model: {self.arbitration_model}")

    def verify_page(
        self,
        image_path: Path,
        extracted_data: dict,
    ) -> Tuple[dict, VerificationSummary]:
        """
        Run full verification pipeline on a page.

        Args:
            image_path: Path to source image
            extracted_data: Dict with 'lab_results' key

        Returns:
            Tuple of (verified_data, summary)
        """
        primary_results = extracted_data.get("lab_results", [])
        summary = VerificationSummary(total_results=len(primary_results))

        if not primary_results:
            logger.info(f"No results to verify for {image_path.name}")
            return extracted_data, summary

        logger.info(f"[{image_path.name}] Starting verification of {len(primary_results)} results")

        # === Stage 1: Cross-Model Extraction ===
        logger.info(f"[{image_path.name}] Stage 1: Cross-model extraction with {self.verification_model}")
        verification_results = cross_model_extract(
            image_path, self.client, self.verification_model
        )
        logger.info(f"[{image_path.name}] Verification model extracted {len(verification_results)} results")

        # === Stage 2: Compare Extractions ===
        logger.info(f"[{image_path.name}] Stage 2: Comparing extractions")
        matching, disagreements, only_in_verification = compare_extractions(
            primary_results, verification_results
        )

        logger.info(f"[{image_path.name}] Comparison: {len(matching)} match, "
                   f"{len(disagreements)} disagree, {len(only_in_verification)} only in verification")

        # Track verification status for each result
        verification_map = {}  # lab_name_raw -> verification info

        # Mark matching results as verified
        for match in matching:
            name = match["primary"].get("lab_name_raw", "")
            verification_map[name] = {
                "status": "verified",
                "confidence": 0.95,
                "method": "cross_model_match",
                "cross_model_value": str(match["verification"].get("value_raw", "")),
            }
            summary.verified_count += 1

        # === Stage 3: Verify Disagreements ===
        if disagreements:
            logger.info(f"[{image_path.name}] Stage 3: Verifying {len(disagreements)} disagreements")

            # Batch verification first
            disagree_results = [d["primary"] for d in disagreements if d["primary"]]
            batch_verifications = verify_values_batch(
                image_path, disagree_results, self.client, self.verification_model
            )

            # Build verification lookup
            batch_lookup = {
                v.get("lab_name_raw", "").lower().strip(): v
                for v in batch_verifications
            }

            for disagree in disagreements:
                primary = disagree["primary"]
                name = primary.get("lab_name_raw", "")
                name_key = name.lower().strip()
                p_val = disagree["primary_value"]
                v_val = disagree["verification_value"]

                batch_result = batch_lookup.get(name_key, {})

                if batch_result.get("value_matches") == "match":
                    # Batch verification confirms primary
                    verification_map[name] = {
                        "status": "verified",
                        "confidence": 0.9,
                        "method": "batch_verification",
                        "actual_value": batch_result.get("actual_value_seen"),
                    }
                    summary.verified_count += 1

                elif batch_result.get("value_matches") == "mismatch":
                    actual = batch_result.get("actual_value_seen")

                    # Check if verification model was correct
                    if actual and v_val and actual.strip() == v_val.strip():
                        verification_map[name] = {
                            "status": "corrected",
                            "confidence": 0.85,
                            "method": "verification_model_correct",
                            "original_value": p_val,
                            "corrected_value": v_val,
                            "actual_value": actual,
                        }
                        summary.corrected_count += 1
                    else:
                        # Need arbitration
                        verification_map[name] = {
                            "status": "needs_arbitration",
                            "primary_value": p_val,
                            "verification_value": v_val,
                            "batch_value": actual,
                        }
                else:
                    # Uncertain - needs character-level verification
                    verification_map[name] = {
                        "status": "uncertain",
                        "primary_value": p_val,
                        "verification_value": v_val,
                    }

        # === Stage 4: Character-Level Verification ===
        if self.enable_character_verification:
            uncertain_names = [
                name for name, info in verification_map.items()
                if info.get("status") in ("uncertain", "needs_arbitration")
            ]

            if uncertain_names:
                logger.info(f"[{image_path.name}] Stage 4: Character-level verification for {len(uncertain_names)} items")

                for name in uncertain_names[:10]:  # Limit to avoid excessive API calls
                    info = verification_map[name]
                    p_val = info.get("primary_value", "")

                    char_result = verify_value_character_level(
                        image_path, name, p_val,
                        self.client, self.arbitration_model
                    )

                    if char_result.get("test_found") and char_result.get("confidence", 0) > 0.7:
                        actual = char_result.get("full_value", "")
                        matches = char_result.get("matches_extracted", False)

                        if matches:
                            verification_map[name] = {
                                "status": "verified",
                                "confidence": char_result.get("confidence", 0.9),
                                "method": "character_level",
                                "actual_value": actual,
                            }
                            summary.verified_count += 1
                        else:
                            verification_map[name] = {
                                "status": "corrected",
                                "confidence": char_result.get("confidence", 0.85),
                                "method": "character_level_correction",
                                "original_value": p_val,
                                "corrected_value": actual,
                            }
                            summary.corrected_count += 1
                    else:
                        # Still uncertain
                        verification_map[name]["confidence"] = 0.5
                        summary.uncertain_count += 1

        # === Stage 5: Arbitration for Remaining Disagreements ===
        arbitration_needed = [
            (name, info) for name, info in verification_map.items()
            if info.get("status") == "needs_arbitration"
        ]

        if arbitration_needed:
            logger.info(f"[{image_path.name}] Stage 5: Arbitrating {len(arbitration_needed)} disagreements")

            for name, info in arbitration_needed[:5]:  # Limit to control costs
                arb_result = arbitrate_disagreement(
                    image_path, name,
                    info.get("primary_value", ""),
                    info.get("verification_value", ""),
                    self.client, self.arbitration_model
                )

                if arb_result.get("confidence", 0) > 0.7:
                    correct_source = arb_result.get("correct_source")
                    actual = arb_result.get("actual_value")

                    if correct_source == "A":
                        verification_map[name] = {
                            "status": "verified",
                            "confidence": arb_result.get("confidence", 0.85),
                            "method": "arbitration_primary",
                            "actual_value": actual,
                        }
                        summary.verified_count += 1
                    elif correct_source == "B":
                        verification_map[name] = {
                            "status": "corrected",
                            "confidence": arb_result.get("confidence", 0.85),
                            "method": "arbitration_correction",
                            "original_value": info.get("primary_value"),
                            "corrected_value": actual or info.get("verification_value"),
                        }
                        summary.corrected_count += 1
                    else:
                        # Neither - flag for review
                        verification_map[name] = {
                            "status": "uncertain",
                            "confidence": 0.4,
                            "method": "arbitration_uncertain",
                            "actual_value": actual,
                            "notes": arb_result.get("reasoning", ""),
                        }
                        summary.uncertain_count += 1

        # === Stage 6: Completeness Check ===
        missed_results = []
        if self.enable_completeness_check:
            logger.info(f"[{image_path.name}] Stage 6: Completeness check")
            missed_results = check_completeness(
                image_path, primary_results, self.client, self.verification_model
            )

            if missed_results:
                logger.warning(f"[{image_path.name}] Found {len(missed_results)} potentially missed results")
                summary.missed_count = len(missed_results)

        # === Apply Verification Results ===
        verified_results = []
        for result in primary_results:
            result = result.copy()
            name = result.get("lab_name_raw", "")

            if name in verification_map:
                vinfo = verification_map[name]

                # Apply corrections
                if vinfo.get("status") == "corrected":
                    result["value_raw_original"] = result.get("value_raw")
                    result["value_raw"] = vinfo.get("corrected_value")
                    result["verification_corrected"] = True

                # Add verification metadata
                result["verification_status"] = vinfo.get("status", "unknown")
                result["verification_confidence"] = vinfo.get("confidence", 0.5)
                result["verification_method"] = vinfo.get("method", "")
                result["cross_model_verified"] = vinfo.get("status") in ("verified", "corrected")

                # Flag for review if uncertain
                if vinfo.get("status") == "uncertain" or vinfo.get("confidence", 1.0) < 0.7:
                    result["needs_review"] = True
                    existing_reason = result.get("review_reason", "") or ""
                    if "VERIFICATION_UNCERTAIN" not in existing_reason:
                        result["review_reason"] = existing_reason + "VERIFICATION_UNCERTAIN; "
            else:
                # Not in verification map (should be rare)
                result["verification_status"] = "not_verified"
                result["verification_confidence"] = 0.6

            verified_results.append(result)

        # Add missed results
        for missed in missed_results:
            missed["verification_status"] = "recovered"
            missed["verification_confidence"] = 0.7
            missed["verification_method"] = "completeness_check"
            missed["needs_review"] = True
            missed["review_reason"] = "RECOVERED_FROM_COMPLETENESS_CHECK; "
            verified_results.append(missed)

        # Calculate average confidence
        confidences = [r.get("verification_confidence", 1.0) for r in verified_results]
        summary.avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        summary.total_results = len(verified_results)

        extracted_data["lab_results"] = verified_results

        logger.info(f"[{image_path.name}] Verification complete: "
                   f"{summary.verified_count} verified, {summary.corrected_count} corrected, "
                   f"{summary.uncertain_count} uncertain, {summary.missed_count} recovered")

        return extracted_data, summary


# ========================================
# Convenience Function
# ========================================

def verify_page_extraction(
    image_path: Path,
    extracted_data: dict,
    client: OpenAI,
    primary_model: str,
    verification_model: Optional[str] = None,
    enable_completeness_check: bool = True,
    enable_character_verification: bool = True,
) -> Tuple[dict, dict]:
    """
    Convenience function to verify a page extraction.

    Args:
        image_path: Path to source image
        extracted_data: Dict with 'lab_results' key
        client: OpenAI client for OpenRouter
        primary_model: Model used for primary extraction
        verification_model: Override verification model
        enable_completeness_check: Check for missed results
        enable_character_verification: Do character-level verification

    Returns:
        Tuple of (verified_data, summary_dict)
    """
    verifier = ExtractionVerifier(
        client=client,
        primary_model=primary_model,
        verification_model=verification_model,
        enable_completeness_check=enable_completeness_check,
        enable_character_verification=enable_character_verification,
    )

    verified_data, summary = verifier.verify_page(image_path, extracted_data)
    return verified_data, summary.to_dict()
