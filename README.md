# labs-parser

[Logo](logo.jpg)

A Python tool that extracts and processes laboratory test results from medical documents using Claude 3 AI model. The tool can:
- Extract structured data from PDF lab reports
- Process multiple documents in parallel
- Generate time series visualizations for lab results
- Merge results into a standardized CSV format

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

To set up the conda environment using the `environment.yml` file, run the following command:

```sh
conda env create -f environment.yml
```

This will create a new conda environment with all the dependencies specified in the `environment.yml` file.

## Usage

1. Create and configure environment file:
   ```sh
   cp .env.example .env
   ```
   Then edit `.env` with your specific configuration:
   ```
   INPUT_PATH=path/to/input/pdfs
   OUTPUT_PATH=path/to/output
   ANTHROPIC_API_KEY=your-api-key
   INPUT_FILE_REGEX=.*analises.*\.pdf$  # Optional
   ```

2. Run the parser:
   ```sh
   python main.py
   ```

3. Check output directory for:
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

```sh
python -m tests.test_pipeline
```

## TODO

- [ ] Skip generated files (csv, txt, etc) if they already exist