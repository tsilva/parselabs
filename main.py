from dotenv import load_dotenv
load_dotenv()

import logging
import os
import json
import shutil
import anthropic
import base64
import pandas as pd
import pdf2image
import re
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from multiprocessing import Pool, cpu_count
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('labs-parser')

with open("config/lab_names.json", "r") as f: LAB_NAMES = json.load(f)
with open("config/lab_methods.json", "r") as f: LAB_METHODS = json.load(f)
with open("config/lab_units.json", "r") as f: LAB_UNITS = json.load(f)

TOOLS = [
    {
        "name": "extract_lab_results",
        "description": f"""Extract structured laboratory test results from medical documents with high precision.
Specific requirements:
1. Extract EVERY test result visible in the image, including variants with different units
2. Use N/A for missing methods, never leave blank or use nan
3. For numeric ranges:
   - Use 0 when no minimum is specified
   - Use 9999 when no maximum is specified
4. Dates must be in ISO 8601 format (YYYY-MM-DD)
5. Units must match exactly as shown in the document""",
        "input_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "Lista de resultados de exames laboratoriais",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Data de realizacao dos exames, em formato ISO 8601 (por exemplo, '2022-01-01'): N/A caso a data nao esteja especificada no documento",
                            },
                            "lab_name": {
                                "type": "string",
                                "enum": LAB_NAMES,
                                "description": "Nome do exame laboratorial"
                            },
                            "lab_method" : {
                                "type": "string",
                                "enum": LAB_METHODS,
                                "description": "Método de medição do resultado do exame laboratorial; N/A para resultados sem método especificado"
                            },
                            "lab_value": {
                                "type": "number",
                                "description": "Valor numérico medido do resultado do exame laboratorial"
                            },
                            "lab_unit": {
                                "type": "string",
                                "enum": LAB_UNITS,
                                "description": "Unidade de medida para o resultado do exame; N/A para resultados sem unidade"
                            },
                            "lab_range_min": {
                                "type": "number",
                                "description": "Limite inferior do intervalo de referência normal. Use 0 se nenhum limite mínimo for especificado."
                            },
                            "lab_range_max": {
                                "type": "number",
                                "description": "Limite superior do intervalo de referência normal. Use 9999 se nenhum limite máximo for especificado."
                            },
                            "confidence" : {
                                "type": "number",
                                "description": "Confiança do modelo na extração fidedigna do resultado do exame laboratorial; entre 0 e 1"
                            },
                            "lack_of_confidence_reason": {
                                "type": "string",
                                "description": "Motivo para a falta de confiança no resultado do exame laboratorial; opcional, fornecido apenas se a confiança for menor que 1"
                            }
                        },
                        "required": [
                            "date",
                            "lab_name",
                            "lab_method",
                            "lab_value",
                            "lab_unit",
                            "lab_range_min",
                            "lab_range_max",
                            "confidence"
                        ]
                    }
                },
                "confidence": {
                    "type": "number",
                    "description": "Confiança geral do modelo na extração fidedigna de todos os resultados de exames laboratoriais; entre 0 e 1"
                },
                "lack_of_confidence_reason": {
                    "type": "string",
                    "description": "Motivo para a falta de confiança em um ou mais resultados de exames laboratoriais; opcional, fornecido apenas se a confiança for menor que 1"
                }
            },
            "required": ["results", "confidence"]
        }
    }
]

def load_env_config():
    # Read environment variables
    input_path = os.getenv("INPUT_PATH")
    input_file_regex = os.getenv("INPUT_FILE_REGEX")
    output_path = os.getenv("OUTPUT_PATH")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Validate required fields
    if not input_path or not Path(input_path).exists(): raise ValueError(f"INPUT_PATH not set or does not exist: {input_path}")
    if not input_file_regex: raise ValueError("INPUT_FILE_REGEX not set")
    if not output_path or not Path(output_path).exists(): raise ValueError("OUTPUT_PATH not set")
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not set")

    return {
        "input_path" : Path(input_path),
        "input_file_regex" : input_file_regex,
        "output_path" : Path(output_path),
        "anthropic_api_key" : anthropic_api_key
    }

def transcription_from_page_image(image_path, client):
    """Get a verbatim transcription of the lab report"""
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    message = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=8192,
        temperature=0.0,
        system=[
            {
                "type": "text",
                "text": """You are a precise document transcriber. Your task is to:
1. Write out ALL text visible in the image exactly as it appears
2. Preserve the document's layout and formatting as much as possible using spaces and newlines
3. Include ALL numbers, units, and reference ranges exactly as shown
4. Use the exact same text case (uppercase/lowercase) as the document
5. Do not interpret, summarize, or structure the content - just transcribe it""",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please transcribe this lab report exactly as it appears, preserving layout and all details. Write the text exactly as shown in the document."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text

def extract_labs_from_page_transcription(transcription, client):
    # Extract structured data from transcription
    message = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=8192,
        temperature=0.0,
        system=[
            {
                "type": "text",
                "text": """You are a medical lab report analyzer with the following strict requirements:
1. COMPLETENESS: Extract ALL test results from the provided transcription
2. ACCURACY: Values and units must match exactly
3. CONSISTENCY: Use N/A for missing methods, never leave blank
4. VALIDATION: Verify each extraction matches the source text exactly
5. THOROUGHNESS: Process the text line by line to ensure nothing is missed""",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Here is the verbatim transcription of a lab report. Extract all test results from this text:

{transcription}

Extract ALL lab results following these steps:
1. Read through the entire text carefully
2. For each test result found:
   - Copy the test name exactly as written
   - Note the method if specified (use N/A if not)
   - Copy the numeric value exactly
   - Copy the unit exactly as shown
   - Record the reference ranges (use 0/9999 if not specified)
3. Verify each extraction against the source text
4. Double-check that no results were missed"""
                    }
                ]
            }
        ],
        tools=TOOLS
    )

    # Process response
    tool_result = None
    for content in message.content:
        if not hasattr(content, "input"): continue
        assert tool_result is None, "Multiple tools detected in message"
        tool_result = content.input
        
    return tool_result

def hash_file(file_path, length=4):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    full_hash = hash_md5.hexdigest()
    short_hash = full_hash[:length]
    return short_hash

def preprocess_page_image(image):
    """Optimize image by converting to grayscale, resizing if needed, and quantizing"""

    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Only resize if image is wider than maximum width
    MAX_WIDTH = 800
    if gray_image.width > MAX_WIDTH:
        # Calculate new height maintaining aspect ratio
        ratio = MAX_WIDTH / gray_image.width
        height = int(gray_image.height * ratio)
        # Resize image
        gray_image = gray_image.resize((MAX_WIDTH, height), Image.Resampling.LANCZOS)
    
    # Quantize to 128 colors and convert back to grayscale for JPEG compatibility
    quantized = gray_image.quantize(colors=128)
    final_image = quantized.convert('L')
    return final_image

def create_lab_test_plot(df_test, lab_name, output_dir):
    """Create a time series plot for a specific lab test"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_test, x='date', y='lab_value', marker='o')
    plt.title(f'Time Series of {lab_name}')
    plt.xlabel('Date')
    plt.ylabel('Lab Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_file = output_dir / f"{lab_name}.png"
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Saved plot for {lab_name} to {output_file}")

class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    def __init__(self, config):
        self.config = config

        # Initialize logger for this step
        step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"{step_name}")

        # Add prefix to all log messages for this step
        for handler in self.logger.handlers:
            handler.setFormatter(
                logging.Formatter(
                    f'%(asctime)s - {step_name} - %(levelname)s - %(message)s'
                )
            )
    
    @abstractmethod
    def execute(self, data: dict) -> dict:
        pass

def _StepCopyPDFs_worker_fn(args):
    """Copy PDF file to destination directory"""
    
    # Read worker args
    pdf_path, output_path = args

    # Create output directory for document assets
    output_dir = output_path / pdf_path.stem
    output_dir.mkdir(exist_ok=True)
    
    # Copy PDF file to output directory
    output_pdf_path = output_dir / pdf_path.name
    shutil.copy2(pdf_path, output_pdf_path)
                 
    # Hash file
    pdf_hash = hash_file(output_pdf_path)

    # Return output path and hash
    return output_pdf_path, pdf_hash

class StepCopyPDFs(PipelineStep):
    """Step 1: Copy PDFs to destination"""
    def execute(self, data: dict) -> dict:        
        # Read configuration
        input_path = self.config["input_path"]
        input_file_regex = self.config["input_file_regex"]
        output_path = self.config["output_path"]
        n_workers = self.config.get("n_workers", cpu_count())

        # Collect file paths for extraction using regex pattern
        input_pdf_paths = [f for f in input_path.iterdir() if re.search(input_file_regex, str(f), re.IGNORECASE)]

        # Limit number of workers to number of input PDFs
        n_workers = min(n_workers, len(input_pdf_paths))

        # Copy PDFs to output directory in parallel
        pdf_hashes_map = {}
        output_pdf_paths = []
        with Pool(n_workers) as pool:
            results = pool.map(_StepCopyPDFs_worker_fn, [(path, output_path) for path in input_pdf_paths])
            for (path, file_hash) in results: 
                pdf_hashes_map[file_hash] = path.name
                output_pdf_paths.append(path)

        # Save registry of file hashes for quick 
        # lookup of existing files in future runs
        registry_path = output_path / "hashes.json"
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_hashes_map, f, indent=2, ensure_ascii=False)
        
        # Return pipeline step output
        return {
            "input_pdf_paths": output_pdf_paths
        }

def _StepExtractPageImages_worker_fn(args):
    """Extract pages from a single PDF with error handling"""
    pdf_path, output_dir, logger = args  # Add logger to args
    try:
        images = pdf2image.convert_from_path(pdf_path)
        base_name = pdf_path.stem
        image_paths = []
        
        for i, image in enumerate(images, start=1):
            processed_image = preprocess_page_image(image)
            image_path = output_dir / f"{base_name}.{i:03d}.jpg"
            processed_image.save(image_path, "JPEG", quality=80)
            image_paths.append(image_path)
        
        logger.info(f"Extracted {len(image_paths)} pages from {pdf_path}")
        return image_paths
    except Exception as e:
        logger.error(f"Error extracting pages from {pdf_path}: {e}", exc_info=True)
        return []

class StepExtractPageImages(PipelineStep):
    def execute(self, data: dict) -> dict:
        
        # Read configuration
        n_workers = self.config.get("n_workers", cpu_count())
        
        # Get input data
        input_pdf_paths = data["input_pdf_paths"]

        # Limit number of workers to number of input PDFs
        n_workers = min(n_workers, len(input_pdf_paths))
        
        # Extract pages in parallel
        pdf_page_image_paths = []
        with Pool(n_workers) as pool:
            args = [(path, path.parent, self.logger) for path in input_pdf_paths]
            results = pool.map(_StepExtractPageImages_worker_fn, args)
            for paths in results: pdf_page_image_paths.extend(paths)
                
        self.logger.info(f"Total pages extracted: {len(pdf_page_image_paths)}")

        # Return pipeline step output
        return {
            "pdf_page_image_paths": pdf_page_image_paths
        }

def _StepExtractPageImageLabs_worker_fn(args):
    """Process single image with Claude"""
    image_path, api_key, output_dir, logger = args
    try:
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)
        
        txt_path = output_dir / f"{image_path.stem}.txt"
        if not txt_path.exists():
            # Transcribe image verbatim
            transcription = transcription_from_page_image(image_path, client)
            
            # Save transcription
            txt_path.write_text(transcription, encoding='utf-8')
            logger.info(f"Saved transcription to {txt_path}")
        else:
            # Load existing transcription
            transcription = txt_path.read_text(encoding='utf-8')
            logger.info(f"Loaded existing transcription from {txt_path}")
            
        # Extract structured data from transcription
        extracted_data = extract_labs_from_page_transcription(transcription, client)
        
        # Log warnings for low confidence
        confidence = extracted_data["confidence"]
        lack_of_confidence_reason = extracted_data.get("lack_of_confidence_reason")
        if confidence < 1:
            logger.warning(f"Confidence below 1 for {image_path}: {confidence}")
            if lack_of_confidence_reason: logger.warning(f" - Lack of confidence reason: {lack_of_confidence_reason}")
        
        # Save structured results
        json_path = output_dir / f"{image_path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extraction results to {json_path}")

        # Process structured data
        labs = []
        results = extracted_data["results"]
        for result in results: labs.append(result)
        labs_df = pd.DataFrame(labs)
        labs_df['date'] = pd.to_datetime(labs_df['date'])
        labs_df = labs_df[[
            "date",
            "lab_name",
            "lab_method",
            "lab_value",
            "lab_unit",
            "lab_range_min",
            "lab_range_max",
            "confidence",
            "lack_of_confidence_reason",
        ]]

        # Normalize N/A values
        labs_df = labs_df.replace({pd.NA: 'N/A', 'nan': 'N/A', pd.NaT: 'N/A'})
        labs_df = labs_df.fillna('N/A')

        # Add source file name
        labs_df['source_file'] = image_path.parent.name + ".pdf"
        
        # Save structured results
        csv_path = output_dir / f"{image_path.stem}.csv"
        labs_df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved page results to {csv_path}")
        
        return image_path, labs_df
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return image_path, pd.DataFrame()

class StepExtractPageImageLabs(PipelineStep):
    def execute(self, data: dict) -> dict:
        # Read configuration
        api_key = self.config["anthropic_api_key"]
        n_workers = self.config.get("n_workers", cpu_count())
        
        # Read previous step's output
        pdf_page_image_paths = data["pdf_page_image_paths"]
        
        # Limit number of workers to number of input images
        n_workers = min(n_workers, len(pdf_page_image_paths))

        # Process images in parallel
        with Pool(n_workers) as pool:
            args = [(path, api_key, path.parent, self.logger) for path in pdf_page_image_paths]
            results = pool.map(_StepExtractPageImageLabs_worker_fn, args)
        
        # Group results by PDF
        pdf_results = {}
        for img_path, labs in results:
            pdf_path = img_path.parent / f"{img_path.parent.name}.pdf"
            if pdf_path not in pdf_results: pdf_results[pdf_path] = []
            pdf_results[pdf_path].append(labs)
        
        # Save aggregated results
        all_results = []
        for pdf_path, labs_list in pdf_results.items():
            if labs_list:
                # Concatenate all dataframes from this PDF
                pdf_df = pd.concat(labs_list, ignore_index=True)
                csv_path = pdf_path.with_suffix('.csv')
                pdf_df.to_csv(csv_path, index=False, sep=';')
                all_results.append(pdf_df)
                self.logger.info(f"Saved page results to {csv_path}")
        
        # Return pipeline step output
        return {
            "results": pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        }

def _StepMergeResults_worker_fn(args):
    """Process single CSV file"""
    csv_path, logger = args
    try:
        df = pd.read_csv(csv_path, sep=';')
        if len(df.columns) <= 1:  # Skip if file is malformed
            logger.warning(f"Skipping malformed CSV: {csv_path}")
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df[[
            "date",
            "lab_name",
            "lab_method",
            "lab_value",
            "lab_unit",
            "lab_range_min",
            "lab_range_max",
            "source_file",
            "confidence",
            "lack_of_confidence_reason",
        ]]
        return df
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}", exc_info=True)
        return None

class StepMergePageLabs(PipelineStep):
    def execute(self, data: dict) -> dict:
        # Read configuration
        output_path = self.config["output_path"]
        n_workers = self.config.get("n_workers", cpu_count())

        # Find CSVs to merge
        csv_files = [f for f in output_path.glob("**/*.csv") 
                    if not re.search(r'\.\d{3}\.csv$', str(f))
                    and f.name != "merged_results.csv"]
        
        # Merge CSVs in parallel
        with Pool(n_workers) as pool:
            results = pool.map(_StepMergeResults_worker_fn, [(f, self.logger) for f in csv_files])
            dfs = [df for df in results if df is not None]
        
        if not dfs:
            self.logger.warning("No CSV files found to merge")
            return {}
            
        # Process merged results
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Convert dates to consistent format
        merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
        
        # Find most frequent date
        most_frequent_date = merged_df['date'].mode().iloc[0]
        self.logger.info(f"Using most frequent date: {most_frequent_date}")
        
        # Set all dates to the most frequent date
        merged_df['date'] = pd.to_datetime(most_frequent_date)
        
        # Sort results
        merged_df = merged_df.sort_values(['lab_name'], ascending=[False])
        
        # Convert date to string format before saving
        merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
        
        # Save results
        output_file = output_path / "merged_results.csv"
        merged_df.to_csv(output_file, index=False, sep=';')
        self.logger.info(f"Saved merged results to {output_file}")
        
        return {"merged_results": merged_df}

class StepPlotLabs(PipelineStep):
    def execute(self, data: dict) -> dict:
        # Read configuration
        output_path = self.config["output_path"]

        # Ensure output directories exist
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        input_path = output_path / "merged_results.csv"
        df = pd.read_csv(input_path, sep=';')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Process each unique lab test
        for lab_name in df['lab_name'].unique():
            df_test = df[df['lab_name'] == lab_name]
            
            # Skip if only one data point
            if len(df_test) <= 1:
                self.logger.info(f"Skipping plot for {lab_name} - only {len(df_test)} data point(s)")
                continue
                
            self.logger.info(f"Creating plot for {lab_name} with {len(df_test)} data points")
            create_lab_test_plot(df_test, lab_name, plots_dir)

        return data

class Pipeline:
    """Main pipeline executor"""
    def __init__(self, steps):
        self.logger = logging.getLogger(f"{__name__}.Pipeline")
        self.steps = steps

    def execute(self):
        data = {}
        for step in self.steps: 
            _data = step.execute(data) or {}
            data = {**data, **_data}
        return data

def create_default_pipeline(plot_labs=True):
    env_config = load_env_config()
    steps = [x for x in [
        StepCopyPDFs({**env_config}),
        StepExtractPageImages({**env_config}),
        StepExtractPageImageLabs({**env_config, "n_workers": 2}),
        StepMergePageLabs({**env_config}),
        StepPlotLabs({**env_config}) if plot_labs else None
    ] if x is not None]
    pipeline = Pipeline(steps)
    return pipeline

def main():
    pipeline = create_default_pipeline()
    pipeline.execute()

if __name__ == "__main__":
    main()
