from dotenv import load_dotenv
load_dotenv()

import logging
import os
import json
import shutil
import anthropic
import base64
from pathlib import Path
import pandas as pd
import pdf2image
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import hashlib
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from abc import ABC, abstractmethod
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger('labs-parser')

with open("config/lab_names.json", "r") as f: LAB_NAMES = json.load(f)
with open("config/lab_methods.json", "r") as f: LAB_METHODS = json.load(f)
with open("config/lab_units.json", "r") as f: LAB_UNITS = json.load(f)

TOOLS = [
    {
        "name": "extract_lab_results",
        "description": f"""Extrair resultados estruturados de exames laboratoriais a partir de documentos médicos. 
Para testes sem um limite mínimo especificado, use 0. 
Para testes sem um limite máximo especificado, use 9999.""",
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
                                "description": "Data em formato ISO 8601 (por exemplo, '2022-01-01')",
                            },
                            "lab_name": {
                                "type": "string",
                                "enum": LAB_NAMES,
                                "description": "Nome do exame laboratorial (por exemplo, 'Hemoglobina', 'Contagem de Glóbulos Brancos')"
                            },
                            "lab_method" : {
                                "type": "string",
                                "enum": LAB_METHODS,
                                "description": "Método de medição do resultado do exame laboratorial (exemplo: 'Imunoensaio', 'Citometria de Fluxo'); N/A para resultados sem método"
                            },
                            "lab_value": {
                                "type": "number",
                                "description": "Valor numérico medido do resultado do exame laboratorial"
                            },
                            "lab_unit": {
                                "type": "string",
                                "enum": LAB_UNITS,
                                "description": "Unidade de medida para o resultado do exame (por exemplo, 'g/dL', 'células/µL'); N/A para resultados sem unidade"
                            },
                            "lab_range_min": {
                                "type": "number",
                                "description": "Limite inferior do intervalo de referência normal. Use 0 se nenhum limite mínimo for especificado."
                            },
                            "lab_range_max": {
                                "type": "number",
                                "description": "Limite superior do intervalo de referência normal. Use 9999 se nenhum limite máximo for especificado."
                            }
                        },
                        "required": [
                            "lab_name",
                            "lab_method",
                            "lab_value",
                            "lab_unit",
                            "lab_range_min",
                            "lab_range_max"
                        ]
                    }
                }
            },
            "required": ["results"]
        }
    }
]

def extract_labs_from_page_image(image_path, client):
    """Process a single image with Claude"""
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        system="You are a meticulous medical lab report analyzer. Extract ALL laboratory test results from this image - missing even one result is considered a failure.",
        messages=[
            {
                "role": "user",
                "content": [
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
        ],
        tools=TOOLS
    )

    labs = []
    for content in message.content:
        if not hasattr(content, "input"): continue
        results = content.input["results"]
        for result in results: labs.append(result)
    return labs

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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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
        self.logger.info("Stage 1: Copying PDFs to destination directories")
        
        # Read configuration
        input_path = self.config["input_path"]
        output_path = self.config["output_path"]

        # Collect file paths for extraction
        # TODO: softcode regex
        input_pdf_paths = [f for f in input_path.glob("*.pdf") if "analises" in f.name.lower()]

        # Calculate number of parallel workers to use in this step
        # (use all available CPU cores)
        n_workers = max(len(input_pdf_paths), cpu_count())

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
            "input_pdf_paths": input_pdf_paths
        }

# Step 2: Extract Pages
def _StepExtractPages_worker_fn(args):
    """Extract pages from a single PDF with error handling"""
    pdf_path, output_dir = args
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

class StepExtractPages(PipelineStep):
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 2: Extracting PDF pages")
        
        # Read configuration
        output_path = self.config["output_path"]
        n_workers = min(self.config["n_workers"], cpu_count())
        
        # Get input data
        input_pdf_paths = data["input_pdf_paths"]
        
        # Extract pages in parallel
        all_image_paths = []
        with Pool(n_workers) as pool:
            args = [(path, path.parent) for path in input_pdf_paths]
            results = pool.map(_StepExtractPages_worker_fn, args)
            for paths in results:
                all_image_paths.extend(paths)
                
        self.logger.info(f"Total pages extracted: {len(all_image_paths)}")
        return {"pdf_page_image_paths": all_image_paths}

# Step 3: Process Images
def _StepProcessImages_worker_fn(args):
    """Process single image with Claude"""
    image_path, api_key, output_dir = args
    try:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Processing {image_path}")
        
        # Extract labs
        labs = extract_labs_from_page_image(image_path, client)
        
        # Save page results
        df = pd.DataFrame(labs)
        csv_path = output_dir / f"{image_path.stem}.csv"
        df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved page results to {csv_path}")
        
        return image_path, labs
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return image_path, []

class StepProcessImages(PipelineStep):
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 3: Processing images")
        
        # Read configuration
        api_key = self.config["anthropic_api_key"]
        output_path = self.config["output_path"]
        n_workers = min(self.config["n_workers"], cpu_count())
        
        # Get input data
        image_paths = data["pdf_page_image_paths"]
        
        # Process images in parallel
        with Pool(n_workers) as pool:
            args = [(path, api_key, path.parent) for path in image_paths]
            results = pool.map(_StepProcessImages_worker_fn, args)
        
        # Group results by PDF
        pdf_results = {}
        for img_path, labs in results:
            pdf_path = img_path.parent / f"{img_path.parent.name}.pdf"
            if pdf_path not in pdf_results:
                pdf_results[pdf_path] = []
            pdf_results[pdf_path].extend(labs)
        
        # Save aggregated results
        all_results = []
        for pdf_path, labs in pdf_results.items():
            if labs:
                df = pd.DataFrame(labs)
                csv_path = pdf_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False, sep=';')
                self.logger.info(f"Saved aggregated results to {csv_path}")
                all_results.extend(labs)
        
        return {"results": all_results}

# Step 4: Merge Results
def _StepMergeResults_worker_fn(args):
    """Process single CSV file"""
    csv_path, = args
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['source_file'] = csv_path.name
        return df
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}", exc_info=True)
        return None

class StepMergeResults(PipelineStep):
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 4: Merging results")
        
        # Read configuration
        output_path = self.config["output_path"]
        n_workers = min(self.config["n_workers"], cpu_count())
        
        # Find CSVs to merge
        csv_files = [f for f in output_path.glob("**/*.csv") 
                    if not re.search(r'\.\d{3}\.csv$', str(f))
                    and f.name != "merged_results.csv"]
        
        # Merge CSVs in parallel
        with Pool(n_workers) as pool:
            results = pool.map(_StepMergeResults_worker_fn, [(f,) for f in csv_files])
            dfs = [df for df in results if df is not None]
        
        if not dfs:
            self.logger.warning("No CSV files found to merge")
            return {}
            
        # Process merged results
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values(['date', 'lab_name'], ascending=[False, False])
        
        # Save results
        output_file = output_path / "merged_results.csv"
        merged_df.to_csv(output_file, index=False, sep=';')
        self.logger.info(f"Saved merged results to {output_file}")
        
        # Save statistics
        self._save_statistics(merged_df, output_path)
        return {"merged_results": merged_df}
        
    def _save_statistics(self, df, output_path):
        """Save statistics and unique values"""
        # ...existing statistics code...

class StepGeneratePlots(PipelineStep):

    def execute(self, data: dict) -> dict:
        # Read configuration
        output_path = self.config["output_path"]

        self.logger.info("Stage 4: Generating plots")

        # Ensure output directories exist
        # DESTINATION_PATH.mkdir(exist_ok=True)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        input_path = output_path / "merged_results.csv"
        df = pd.read_csv(input_path, sep=';')
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Process each unique lab test
        for lab_name in df['lab_name'].unique():
            logger.info(f"Processing {lab_name}")
            df_test = df[df['lab_name'] == lab_name]
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

# TODO: should env vars be read here?
# TODO: is there an easier way to perform env validation?
def main():
    logger.info("Starting pipeline execution")

    # Directory constants
    INPUT_PATH = os.getenv("INPUT_PATH")
    OUTPUT_PATH = os.getenv("OUTPUT_PATH")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Assert input path exists
    if not INPUT_PATH or not Path(INPUT_PATH).exists():
        logger.error(f"Input path does not exist: {INPUT_PATH}")
        sys.exit(1)

    # Assert output path exists
    if not OUTPUT_PATH or not Path(OUTPUT_PATH).exists():
        logger.error(f"Output path does not exist: {OUTPUT_PATH}")
        sys.exit(1)

    # Assert API key is set
    if not ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not found")
        sys.exit(1)

    # Execute pipeline
    base_config = {
        "input_path": Path(INPUT_PATH),
        "output_path": Path(OUTPUT_PATH),
        "anthropic_api_key": ANTHROPIC_API_KEY
    }
    pipeline = Pipeline([
        StepCopyPDFs({**base_config, "n_workers": cpu_count()}),
        StepExtractPages({**base_config, "n_workers": 2}),
        StepProcessImages({**base_config, "n_workers": 2}),
        StepMergeResults({**base_config, "n_workers": 2}),
        StepGeneratePlots({**base_config, "n_workers": 2})
    ])
    pipeline.execute()

if __name__ == "__main__":
    main()
