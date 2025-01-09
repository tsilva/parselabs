# labs-parser

<p align="center">
  <img src="logo.jpg" alt="Labs Parser Logo" width="400"/>
</p>

A Python tool that extracts laboratory test results from medical documents. The tool can:

- Extract structured data from PDF lab reports
- Process multiple documents in parallel
- Merge results into a standardized CSV format
- Generate time series visualizations for lab results

## Project Structure

```
labs-parser/
├── config/               # Configuration files
│   ├── lab_names.json   # Valid laboratory test names
│   ├── lab_methods.json # Valid measurement methods
│   └── lab_units.json   # Valid measurement units
├── tests/               # Test files
└── main.py             # Main pipeline implementation
```

## Installation

1. Set up the conda environment:
   ```sh
   conda env create -f environment.yml
   ```

2. Create and configure environment file:
   ```sh
   cp .env.example .env
   ```
   Then edit `.env` with your configuration. See `.env.example` for required variables and their descriptions.

## Usage

1. Run the parser:
   ```sh
   python main.py
   ```

2. Check output directory for:
   - Extracted page images (JPG)
   - Page transcriptions (TXT)
   - Structured results per page (CSV)
   - Merged results (merged_results.csv)
   - Time series plots (plots/*.png)

## Updating the Environment

If you need to update the conda environment with any changes made to the `environment.yml` file, run:

```sh
conda env update --file environment.yml --prune
```

The `--prune` flag will remove any dependencies that are no longer required.

## Testing

The project includes unit tests to verify the parsing pipeline functionality:

```sh
# Run all tests
python -m tests.test_pipeline

# Run specific test case
python -m tests.test_pipeline TestPipeline.test_extract_labs
```

## TODO

- [ ] Skip generated files (csv, txt, etc) if they already exist