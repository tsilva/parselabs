# 🧪 lab-parser

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

> 🔬 Extract structured lab test results from medical documents with AI precision

Labs Parser is a Python tool that uses Claude AI to extract laboratory test results from medical documents, converting them into structured data for analysis and visualization. It processes images of lab reports, transcribes the text, and extracts standardized test results with high accuracy.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/labs-parser.git
cd labs-parser

# Install with pipx for isolated environment
pipx install . --force
```

## 🛠️ Usage

1. Configure your environment variables in a `.env` file:

```
MODEL_ID=claude-3-7-sonnet-latest
INPUT_PATH=./path/to/lab/reports
INPUT_FILE_REGEX=.*\.jpg
OUTPUT_PATH=./output
ANTHROPIC_API_KEY=your_api_key_here
```

2. Run the parser:

```bash
python main.py
```

3. The tool will:
   - Process each lab report image
   - Transcribe the text content
   - Extract structured lab results
   - Save results as CSV files

## ✨ Features

- Extracts lab test names, values, units, and reference ranges
- Standardizes lab names and units using controlled vocabularies
- Processes multiple documents in parallel
- Caches results to avoid reprocessing
- Generates clean, structured CSV output

## 📊 Output

For each processed document, the tool generates:
- Preprocessed images (JPG)
- Text transcriptions (TXT)
- Structured data (JSON)
- Tabular data (CSV)
- A merged CSV with all results

## 📄 License

This project is licensed under the [MIT License](LICENSE).