# 🧪 lab-parser

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

> 🔬 Extract structured lab test results from medical documents with AI precision

Labs Parser is a Python tool that uses AI (via OpenRouter API) to extract laboratory test results from medical documents, converting them into structured data for analysis and visualization. It processes images of lab reports and extracts standardized test results with high accuracy.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/labs-parser.git
cd labs-parser

# Set up the virtual environment with uv
./activate-env.sh    # or activate-env.bat on Windows
```

## 🛠️ Usage

1. Configure your environment variables in a `.env` file:

```
SELF_CONSISTENCY_MODEL_ID=google/gemini-3-flash-preview
EXTRACT_MODEL_ID=google/gemini-3-flash-preview
INPUT_PATH=./path/to/lab/reports
INPUT_FILE_REGEX=.*\.pdf
OUTPUT_PATH=./output
OPENROUTER_API_KEY=your_api_key_here
N_EXTRACTIONS=3
MAX_WORKERS=1
```

2. Run the parser:

```bash
python extract.py
```

3. The tool will:
   - Process each lab report PDF
   - Extract structured lab results directly from images
   - Standardize lab names and units
   - Save results as CSV and Excel files

## ✨ Features

- Extracts lab test names, values, units, and reference ranges
- Standardizes lab names and units using controlled vocabularies
- **Value validation** detects extraction errors automatically:
  - Biologically impossible values (negative concentrations, out-of-range percentages)
  - Inter-lab relationship mismatches (e.g., LDL vs calculated LDL)
  - Format artifacts (concatenation errors like "52.6=1946")
  - Extreme deviations from reference ranges
- Processes multiple documents in parallel
- Caches results to avoid reprocessing
- Generates clean, structured CSV output with review flags

## 🔍 Review UI

A Streamlit-based interface for reviewing extracted lab results against source documents.

### Installation

```bash
pip install -r review_ui/requirements.txt
```

### Running the Review UI

```bash
streamlit run review_ui/app.py
```

The UI reads from your `OUTPUT_PATH` (configured in `.env` or via environment variable).

### Features

- **Side-by-side view**: Source document image alongside extracted data
- **Keyboard shortcuts**: Y=Accept, N=Reject, S=Skip, Arrow keys=Navigate
- **Filter modes**: Unreviewed, All, Low Confidence, Needs Review, Accepted, Rejected
- **Progress tracking**: Shows review progress with accept/reject counts
- **Persistent storage**: Review status saved directly to extraction JSON files

### Workflow

1. Run `python extract.py` to extract lab results
2. Launch the review UI with `streamlit run review_ui/app.py`
3. Review each extraction:
   - Compare extracted values against the source image
   - Press **Y** to accept correct extractions
   - Press **N** to reject incorrect ones
   - Press **S** to skip and return later
4. Use filters to focus on low-confidence or flagged items

## 🏗️ Architecture

For detailed pipeline documentation, see [docs/pipeline.md](docs/pipeline.md).

## 📊 Output

For each processed document, the tool generates:
- Preprocessed images (JPG)
- Structured data (JSON)
- Tabular data (CSV)
- A merged CSV and Excel file with all results
- Time-series plots for each lab test

## 📄 License

This project is licensed under the [MIT License](LICENSE).
