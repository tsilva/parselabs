# 🧪 labs-parser

<p align="center">
  <img src="logo.jpg" alt="Labs Parser Logo" width="400"/>
</p>

> 🤖 AI-powered lab report parser that turns medical PDFs into structured data

A Python tool that extracts laboratory test results from medical documents with high precision:

🎯 **Key Features**:
- 📄 Extract structured data from PDF lab reports
- ⚡ Process multiple documents in parallel
- 📊 Generate beautiful time series visualizations
- 🔄 Smart caching to avoid reprocessing
- 📁 Clean, standardized CSV output

## 🗂️ Project Structure

```
labs-parser/
├── config/               # Configuration files
│   ├── lab_names.json   # Valid laboratory test names
│   ├── lab_methods.json # Valid measurement methods
│   └── lab_units.json   # Valid measurement units
├── tests/               # Test files
└── main.py             # Main pipeline implementation
```

## ⚙️ Installation

1. Set up the conda environment:
   ```sh
   conda env create -f environment.yml
   ```

2. Create and configure environment file:
   ```sh
   cp .env.example .env
   ```
   Then edit `.env` with your configuration. See `.env.example` for required variables.

## 🚀 Usage

1. Run the parser:
   ```sh
   python main.py
   ```

2. Check output directory for:
   ```
   📂 output/
   ├── 📊 plots/          # Time series visualizations
   ├── 🖼️ *.jpg           # Extracted page images
   ├── 📝 *.txt           # Page transcriptions
   ├── 📑 *.csv           # Structured results
   └── 📈 merged_results.csv
   ```

## 🧪 Testing

Run the test suite to verify functionality:

```sh
# Run all tests 🔍
python -m tests.test_pipeline

# Run specific test case 🎯
python -m tests.test_pipeline TestPipeline.test_extract_labs
```

## 📝 TODO

- [ ] Split transcription into its own step, add progress bar
- [ ] Add progress bar to extraction step
- [ ] Add caching support
- [ ] Make tests use same folder structure as output
- [ ] Bump up number of workers
- [ ] BUG: ferritin plot not working
- [ ] BUG: merged csv should point to file hashes

# Labs Parser Verification Tool

This tool uses Claude AI to verify the accuracy of lab test data extraction from images to CSV files.

## Requirements

- Python 3.7+
- Anthropic API key (for Claude)
- Required Python packages:
  - pandas
  - anthropic
  - tqdm

## Installation

```bash
pip install pandas anthropic tqdm
```

## Usage

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

Run the verification script:

```bash
python test.py --output-dir ./output
```

### Command-line options:

- `--api-key`: Your Anthropic API key (alternatively, set ANTHROPIC_API_KEY environment variable)
- `--output-dir`: Directory to scan for lab test files (default: ./output)
- `--limit`: Limit number of files to process (optional)

## How It Works

1. The script recursively searches the specified directory for matching pairs of .jpg and .csv files
2. For each pair, it:
   - Reads the lab data from the CSV
   - Sends the JPG to Claude for analysis
   - Compares Claude's interpretation with the CSV data
   - Reports any discrepancies

## Output

Results are written to `claude_verification_results.txt` in the output directory. For each file pair, the assessment includes:
- Whether all tests from the image are in the CSV
- List of any missing tests or discrepancies
- Claude's confidence level in the assessment