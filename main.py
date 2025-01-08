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
        n_workers = self.config["n_workers"]

        # TODO: softcode regex
        # Collect file paths for extraction
        pdf_hashes_map = {}
        output_pdf_paths = []
        input_pdf_paths = [f for f in input_path.glob("*.pdf") if "analises" in f.name.lower()]
        n_workers = max(min(n_workers, len(input_pdf_paths)), cpu_count())
        with Pool(n_workers) as pool:
            results = pool.map(_StepCopyPDFs_worker_fn, [(path, output_path) for path in input_pdf_paths])
            for (path, file_hash) in results: 
                pdf_hashes_map[file_hash] = path.name
                output_pdf_paths.append(path)

        # Save hash registry
        registry_path = output_path / "hashes.json"
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_hashes_map, f, indent=2, ensure_ascii=False)
        
        return {
            "input_pdf_paths": input_pdf_paths, 
            "hash_registry": pdf_hashes_map
        }

def _StepExtractPages_worker_fn(args):
    """Extract each page of PDF as an image"""

    # Extract images from PDF
    pdf_path, = args
    images = pdf2image.convert_from_path(pdf_path)

    # Create output path creator function
    base_name = pdf_path.stem
    output_dir = pdf_path.parent
    output_path_fn = lambda x: output_dir / f"{base_name}.{x:03d}.jpg"

    # Process and save each image
    image_paths = []
    for i, image in enumerate(images, start=1):
        processed_image = preprocess_page_image(image)
        image_path = output_path_fn(i)
        processed_image.save(image_path, "JPEG", quality=80)
        image_paths.append(image_path)
    
    # Return paths of extracted images
    return image_paths

class StepExtractPages(PipelineStep):
    """Step 2: Extract pages from PDFs as images"""

    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 2: Extracting PDF pages")

        n_workers = self.config["n_workers"]

        input_pdf_paths = data["input_pdf_paths"]
        n_workers = max(min(n_workers, len(input_pdf_paths)), cpu_count())
        self.logger.info(f"Extracting pages from {len(input_pdf_paths)} PDFs using {n_workers} workers")
        
        pdf_page_image_paths = []
        with Pool(n_workers) as pool:
            results = pool.map(_StepExtractPages_worker_fn, [(path,) for path in input_pdf_paths])
            for paths in results: pdf_page_image_paths.extend(paths)
        self.logger.info(f"Total pages extracted: {len(pdf_page_image_paths)}")

        return {
            "pdf_page_image_paths": pdf_page_image_paths
        }

def _StepProcessImages_worker_fn(args):
    """Process a single page with error handling"""
    image_path, anthropic_api_key = args
    try:
        # Initialize Claude client
        claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        logger.info(f"Processing {image_path}")
        labs = extract_labs_from_page_image(image_path, claude_client)
        df = pd.DataFrame(labs)
        csv_path = image_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved page results to {csv_path}")
        return labs
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return []
            
class StepProcessImages(PipelineStep):
    """Step 3: Process extracted images"""

    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 3: Processing images")

        anthropic_api_key = self.config["anthropic_api_key"]
        n_workers = self.config["n_workers"]

        pdf_page_image_paths = data["pdf_page_image_paths"]

        # Process images in parallel
        n_workers = min(n_workers, len(pdf_page_image_paths))
        self.logger.info(f"Processing {len(pdf_page_image_paths)} images with {n_workers} workers")
        with Pool(n_workers) as pool: results = pool.map(_StepProcessImages_worker_fn, [(path, anthropic_api_key) for path in pdf_page_image_paths])
        
        # Group results by PDF
        pdf_results = {}
        for img_path, labs in zip(pdf_page_image_paths, results):
            pdf_path = img_path.parent / f"{img_path.parent.name}.pdf"
            if pdf_path not in pdf_results:
                pdf_results[pdf_path] = []
            pdf_results[pdf_path].extend(labs)
        
        # Save aggregated results per PDF
        all_results = []
        for pdf_path, labs in pdf_results.items():
            if labs:
                df = pd.DataFrame(labs)
                csv_path = pdf_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False, sep=';')
                self.logger.info(f"Saved aggregated results to {csv_path}")
                all_results.extend(labs)
        
        return {"results": all_results}

class StepMergeResults(PipelineStep):
    """Step 3: Merge all results"""

    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 3: Merging results")

        # Read configuration
        output_path = self.config["output_path"]

        """Merge all CSV files in directory into a single sorted file"""
        # Find all CSV files, excluding page-specific CSVs using regex
        csv_files = [f for f in output_path.glob("**/*.csv") if not re.search(r'\.\d{3}\.csv$', str(f))]
        logger.info(f"Merging {len(csv_files)} CSV files")
        
        # Read and combine all CSVs
        dfs = []
        for csv_file in csv_files:
            if csv_file.name == "merged_results.csv": continue
            df = pd.read_csv(csv_file, sep=';')
            df['source_file'] = csv_file.name
            dfs.append(df)
        
        if not dfs:
            logger.warning("No CSV files found to merge")
            return
        
        # Combine all dataframes and sort
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values(
            by=['date', 'lab_name'], 
            ascending=[False, False]
        )
        
        # Export merged results
        output_path = output_path / "merged_results.csv"
        merged_df.to_csv(output_path, index=False, sep=';')
        logger.info(f"Saved merged results to {output_path}")
        
        # Print statistics and export unique values
        logger.info(f"Total records: {len(merged_df)}")
        logger.info(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
        logger.info(f"Unique lab tests: {len(merged_df['lab_name'].unique())}")
        
        unique_values = {
            "lab_names": sorted([str(x) for x in merged_df['lab_name'].unique().tolist()]),
            "lab_units": sorted([str(x) for x in merged_df['lab_unit'].unique().tolist()]),
            "lab_methods": sorted([str(x) for x in merged_df['lab_method'].unique().tolist()])
        }
        
        for key, values in unique_values.items():
            json_path = output_path / f"unique_{key}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(values, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved unique {key} to {json_path}")

        return data

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
    