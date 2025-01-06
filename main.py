from dotenv import load_dotenv
load_dotenv()

import json
import anthropic
import base64
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Directory constants
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
PLOTS_DIR = Path("plots")

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
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

def extract_pdf_pages(pdf_path):
    """Extract each page of PDF as an image"""
    images = convert_from_path(pdf_path)
    base_name = pdf_path.stem
    image_paths = []
    
    for i, image in enumerate(images, start=1):
        image_path = OUTPUT_DIR / f"{base_name}.{i:03d}.png"
        image.save(image_path, "PNG")
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
                            "media_type": "image/png",
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
    print(f"Split PDF into {len(image_paths)} pages")

    # Process each page
    all_labs = []
    for img_path in image_paths:
        print(f"Processing {img_path}")
        labs = process_image(img_path, client)
        df = pd.DataFrame(labs)
        csv_path = img_path.with_suffix('.csv')
        csv_path = Path(str(csv_path).replace("input", "output"))
        df.to_csv(csv_path, index=False, sep=';')
        print(f"Saved page results to {csv_path}")
        all_labs.extend(labs)

    # Create aggregated results file
    if all_labs:
        df = pd.DataFrame(all_labs)
        csv_path = pdf_path.with_suffix('.csv')
        csv_path = Path(str(csv_path).replace("input", "output"))
        df.to_csv(csv_path, index=False, sep=';')
        print(f"Saved aggregated results to {csv_path}")
    
    return all_labs

def merge_csv_files():
    """Merge all CSV files in directory into a single sorted file"""
    # Find all CSV files, excluding page-specific CSVs using regex
    csv_files = [f for f in OUTPUT_DIR.glob("**/*.csv") 
                 if not re.search(r'\.\d{3}\.csv$', str(f))]
    print(f"\nMerging {len(csv_files)} CSV files")
    
    # Read and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        if csv_file.name == "merged_results.csv":
            continue
        df = pd.read_csv(csv_file, sep=';')
        df['source_file'] = csv_file.name
        dfs.append(df)
    
    if not dfs:
        print("No CSV files found to merge")
        return
    
    # Combine all dataframes and sort
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(
        by=['date', 'lab_name'], 
        ascending=[False, False]
    )
    
    # Export merged results
    output_path = OUTPUT_DIR / "merged_results.csv"
    merged_df.to_csv(output_path, index=False, sep=';')
    print(f"Saved merged results to {output_path}")
    
    # Print statistics and export unique values
    print(f"\nTotal records: {len(merged_df)}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Unique lab tests: {len(merged_df['lab_name'].unique())}")
    
    unique_values = {
        "lab_names": sorted([str(x) for x in merged_df['lab_name'].unique().tolist()]),
        "lab_units": sorted([str(x) for x in merged_df['lab_unit'].unique().tolist()]),
        "lab_methods": sorted([str(x) for x in merged_df['lab_method'].unique().tolist()])
    }
    
    for key, values in unique_values.items():
        json_path = OUTPUT_DIR / f"unique_{key}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
        print(f"Saved unique {key} to {json_path}")

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
    print(f"Saved plot for {lab_name} to {output_file}")

def plot_all_lab_tests():
    """Generate plots for all lab tests from merged results"""
    # Read merged results
    input_path = OUTPUT_DIR / "merged_results.csv"
    df = pd.read_csv(input_path, sep=';')
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Process each unique lab test
    for lab_name in df['lab_name'].unique():
        print(f"Processing {lab_name}")
        df_test = df[df['lab_name'] == lab_name]
        create_lab_test_plot(df_test, lab_name, PLOTS_DIR)

if __name__ == "__main__":
    # Process each PDF file
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    client = anthropic.Anthropic()
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}")
        results = process_pdf(pdf_file, client)
    
    # Merge all results after processing all PDFs
    merge_csv_files()
    
    # Generate plots as final step
    print("\nGenerating plots...")
    plot_all_lab_tests()
    print("Processing complete!")
