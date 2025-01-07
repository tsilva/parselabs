from dotenv import load_dotenv
load_dotenv()

import logging
import os
import json
import anthropic
import base64
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path
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

# Directory constants
SOURCE_PATH = Path(os.getenv("SOURCE_PATH"))
DESTINATION_PATH = Path(os.getenv("DESTINATION_PATH"))
PLOTS_DIR = DESTINATION_PATH / "plots"

# Validate required paths exist
if not SOURCE_PATH.exists():
    logger.error(f"Source path does not exist: {SOURCE_PATH}")
    sys.exit(1)

if not DESTINATION_PATH.exists():
    logger.error(f"Destination path does not exist: {DESTINATION_PATH}")
    sys.exit(1)

# Ensure output directories exist
# DESTINATION_PATH.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

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

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_short_hash(full_hash, length=4):  # Changed from length=8 to length=4
    """Get shortened version of hash"""
    return full_hash[:length]

def setup_hash_directory(pdf_path):
    """Setup hash-based directory structure and return new file path"""
    full_hash = calculate_md5(pdf_path)
    short_hash = get_short_hash(full_hash)
    
    # Create hash-based directory name with original filename
    dir_name = f"[{short_hash}]-{pdf_path.stem}"  # Changed format to use square brackets
    hash_dir = DESTINATION_PATH / dir_name
    hash_dir.mkdir(exist_ok=True)
    
    # New PDF path
    new_pdf_path = hash_dir / f"{dir_name}.pdf"
    
    # Copy file if it doesn't exist
    if not new_pdf_path.exists():
        from shutil import copy2
        copy2(pdf_path, new_pdf_path)
        logger.info(f"Copied {pdf_path.name} to {new_pdf_path}")
    
    return new_pdf_path

def optimize_image(image):
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

def extract_pdf_pages(pdf_path):
    """Extract each page of PDF as an image"""
    images = convert_from_path(pdf_path)
    base_name = pdf_path.stem
    output_dir = pdf_path.parent  # Use the same directory as the PDF
    image_paths = []
    
    for i, image in enumerate(images, start=1):
        # Optimize image before saving
        optimized = optimize_image(image)
        image_path = output_dir / f"{base_name}.{i:03d}.jpg"  # Changed extension to jpg
        optimized.save(image_path, "JPEG", quality=80)  # Save as JPEG with 80% quality
        image_paths.append(image_path)
    
    return image_paths

def process_image(image_path, client):
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

def process_pdf(pdf_path, client):
    """Process a PDF file and extract lab results"""
    # Extract pages as images
    image_paths = extract_pdf_pages(pdf_path)
    logger.info(f"Split PDF into {len(image_paths)} pages")

    # Process each page
    all_labs = []
    for img_path in image_paths:
        logger.info(f"Processing {img_path}")
        labs = process_image(img_path, client)
        df = pd.DataFrame(labs)
        
        # Save CSV in same directory as the source PDF
        csv_path = img_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved page results to {csv_path}")
        all_labs.extend(labs)

    # Create aggregated results file in the same subfolder
    if all_labs:
        df = pd.DataFrame(all_labs)
        csv_path = pdf_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved aggregated results to {csv_path}")
    
    return all_labs

def merge_csv_files():
    """Merge all CSV files in directory into a single sorted file"""
    # Find all CSV files, excluding page-specific CSVs using regex
    csv_files = [f for f in DESTINATION_PATH.glob("**/*.csv") 
                 if not re.search(r'\.\d{3}\.csv$', str(f))]
    logger.info(f"Merging {len(csv_files)} CSV files")
    
    # Read and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        if csv_file.name == "merged_results.csv":
            continue
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
    output_path = DESTINATION_PATH / "merged_results.csv"
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
        json_path = DESTINATION_PATH / f"unique_{key}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved unique {key} to {json_path}")

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

def plot_all_lab_tests():
    """Generate plots for all lab tests from merged results"""
    # Read merged results
    input_path = DESTINATION_PATH / "merged_results.csv"
    df = pd.read_csv(input_path, sep=';')
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Process each unique lab test
    for lab_name in df['lab_name'].unique():
        logger.info(f"Processing {lab_name}")
        df_test = df[df['lab_name'] == lab_name]
        create_lab_test_plot(df_test, lab_name, PLOTS_DIR)

def stage_copy_pdfs():
    """Stage 1: Copy PDFs to hash-based directories"""
    pdf_files = [f for f in SOURCE_PATH.glob("*.pdf") if "analises" in f.name.lower()]
    copied_paths = []
    
    logger.info("Stage 1: Copying PDFs to destination directories")
    for pdf_file in pdf_files:
        hash_pdf_path = setup_hash_directory(pdf_file)
        copied_paths.append(hash_pdf_path)
        logger.info(f"Prepared {hash_pdf_path}")
    
    return copied_paths

def stage_process_pdfs(pdf_paths, client):
    """Stage 2: Process copied PDFs"""
    logger.info("Stage 2: Processing PDFs")
    for pdf_path in pdf_paths:
        logger.info(f"Processing {pdf_path}")
        results = process_pdf(pdf_path, client)

def stage_merge_results():
    """Stage 3: Merge results"""
    logger.info("Stage 3: Merging results")
    merge_csv_files()

def stage_generate_plots():
    """Stage 4: Generate plots"""
    logger.info("Stage 4: Generating plots")
    plot_all_lab_tests()

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    parallel_workers: dict[str, int] = None
    
    def __post_init__(self):
        self.parallel_workers = self.parallel_workers or {}
    
    def get_workers(self, step_name: str, default: int = 1) -> int:
        """Get number of workers for a step"""
        return self.parallel_workers.get(step_name, default)

class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def execute(self, data: dict, pool: Pool = None) -> dict:
        pass

class StepCopyPDFs(PipelineStep):
    """Step 1: Copy PDFs to destination"""
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 1: Copying PDFs to destination directories")
        copied_paths = stage_copy_pdfs()
        return {"pdf_paths": copied_paths}

def extract_single_pdf(args):
    """Extract pages from a single PDF with error handling"""
    logger = logging.getLogger(f"{__name__}.extract_single_pdf")
    pdf_path, = args
    try:
        images = convert_from_path(pdf_path)
        base_name = pdf_path.stem
        output_dir = pdf_path.parent
        image_paths = []
        
        for i, image in enumerate(images, start=1):
            optimized = optimize_image(image)
            image_path = output_dir / f"{base_name}.{i:03d}.jpg"
            optimized.save(image_path, "JPEG", quality=80)
            image_paths.append(image_path)
        
        logger.info(f"Extracted {len(image_paths)} pages from {pdf_path}")
        return image_paths
    except Exception as e:
        logger.error(f"Error extracting pages from {pdf_path}: {e}", exc_info=True)
        return []

class StepExtractPages(PipelineStep):
    """Step 2: Extract pages from PDFs as images"""
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 2: Extracting PDF pages")
        pdf_paths = data["pdf_paths"]
        
        # Prepare arguments for parallel processing
        args = [(path,) for path in pdf_paths]
        
        # Configure parallel processing
        n_workers = self.config.get_workers("extract_pages", cpu_count())
        n_workers = min(n_workers, len(pdf_paths))
        
        all_image_paths = []
        if n_workers > 1:
            self.logger.info(f"Extracting pages from {len(pdf_paths)} PDFs using {n_workers} workers")
            with Pool(n_workers) as pool:
                results = pool.map(extract_single_pdf, args)
                for paths in results:
                    all_image_paths.extend(paths)
        else:
            self.logger.info("Extracting pages sequentially")
            for args in args:
                paths = extract_single_pdf(args)
                all_image_paths.extend(paths)
        
        self.logger.info(f"Total pages extracted: {len(all_image_paths)}")
        return {
            "pdf_paths": pdf_paths,
            "image_paths": all_image_paths
        }

class StepProcessImages(PipelineStep):
    """Step 3: Process extracted images"""
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 3: Processing images")
        image_paths = data["image_paths"]
        client = anthropic.Anthropic()
        
        # Prepare args for parallel processing
        args = [(path, client.api_key) for path in image_paths]
        
        # Process images in parallel
        n_workers = self.config.get_workers("process_images", cpu_count())
        n_workers = min(n_workers, len(image_paths))
        
        if n_workers > 1:
            self.logger.info(f"Processing {len(image_paths)} images with {n_workers} workers")
            with Pool(n_workers) as pool:
                results = pool.map(process_single_page, args)
        else:
            self.logger.info("Processing images sequentially")
            results = [process_single_page(arg) for arg in args]
        
        # Group results by PDF
        pdf_results = {}
        for img_path, labs in zip(image_paths, results):
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
        merge_csv_files()
        return data

class StepGeneratePlots(PipelineStep):
    """Step 4: Generate plots"""
    def execute(self, data: dict) -> dict:
        self.logger.info("Stage 4: Generating plots")
        plot_all_lab_tests()
        return data

def process_single_page(args):
    """Process a single page with error handling"""
    logger = logging.getLogger(f"{__name__}.process_single_page")
    image_path, client_key = args
    try:
        logger.info(f"Processing {image_path}")
        client = anthropic.Anthropic(api_key=client_key)
        labs = process_image(image_path, client)
        df = pd.DataFrame(labs)
        csv_path = image_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved page results to {csv_path}")
        return labs
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return []

class Pipeline:
    """Main pipeline executor"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Pipeline")
        self.steps = [
            StepCopyPDFs(config),
            StepExtractPages(config),
            StepProcessImages(config),
            StepMergeResults(config),
            StepGeneratePlots(config)
        ]
    
    def get_pool_for_step(self, step: PipelineStep, items_count: int) -> Pool:
        step_name = step.__class__.__name__.lower()
        n_workers = self.config.parallel_workers.get(step_name, 1)
        if n_workers > 1:
            n_workers = min(n_workers, items_count)
            self.logger.info(f"Creating pool with {n_workers} workers")
            return Pool(n_workers)
        return None

    def execute(self):
        """Execute all pipeline steps"""
        data = {}
        try:
            for step in self.steps:
                self.logger.info(f"Executing step: {step.__class__.__name__}")
                items_count = len(data.get("pdf_paths", [])) if "pdf_paths" in data else \
                            len(data.get("image_paths", [])) if "image_paths" in data else 1
                
                pool = self.get_pool_for_step(step, items_count)
                if pool:
                    with pool:
                        data = step.execute(data, pool)
                else:
                    data = step.execute(data)
                
                if not data:
                    self.logger.warning("Pipeline stopped: no data to process")
                    break
            self.logger.info("Processing complete!")
        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting pipeline execution")
    # Configure pipeline
    config = PipelineConfig(
        parallel_workers={
            "extract_pages": cpu_count(),
            "process_images": 2
        }
    )
    
    # Execute pipeline
    pipeline = Pipeline(config)
    pipeline.execute()
