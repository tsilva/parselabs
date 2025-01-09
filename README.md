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