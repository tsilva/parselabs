from dotenv import load_dotenv
load_dotenv()

import unittest
from pathlib import Path
import pandas as pd
import tempfile
import shutil
import os
from main import create_default_pipeline

def compare_dataframes(actual_df: pd.DataFrame, expected_df: pd.DataFrame) -> dict:
    """Compare two dataframes and return detailed differences"""
    # Ensure both dataframes have the same columns for comparison
    columns_to_compare = list(set(actual_df.columns) & set(expected_df.columns))
    
    # Create dictionaries keyed by lab_name for easy lookup
    actual_dict = {row['lab_name']: row for _, row in actual_df.iterrows()}
    expected_dict = {row['lab_name']: row for _, row in expected_df.iterrows()}
    
    # Find missing and extra labs
    actual_labs = set(actual_dict.keys())
    expected_labs = set(expected_dict.keys())
    
    missing_labs = expected_labs - actual_labs
    extra_labs = actual_labs - expected_labs
    matching_labs = actual_labs & expected_labs
    
    # Collect missing and extra rows
    missing_rows = [dict(expected_dict[lab]) for lab in missing_labs]
    extra_rows = [dict(actual_dict[lab]) for lab in extra_labs]
    
    # Find differences in matching rows
    matching_diffs = []
    for lab_name in matching_labs:
        actual_row = actual_dict[lab_name]
        expected_row = expected_dict[lab_name]
        
        differences = {
            col: {
                'actual': actual_row[col],
                'expected': expected_row[col]
            }
            for col in columns_to_compare
            if actual_row[col] != expected_row[col]
        }
        
        if differences:
            matching_diffs.append({
                'lab_name': lab_name,
                'differences': differences
            })
    
    return {
        'missing_rows': missing_rows,
        'extra_rows': extra_rows,
        'matching_differences': matching_diffs,
        'total_rows': {
            'actual': len(actual_df),
            'expected': len(expected_df)
        }
    }

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""

        # Create temp dirs (cleaned up after test)
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        #self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

        # Update environment variables (cleaned up after test)
        original_env = dict(os.environ)
        os.environ.update({
            "INPUT_PATH": str(self.input_dir),
            "OUTPUT_PATH": str(self.output_dir),
            "INPUT_FILE_REGEX": r".*\.pdf$"
        })
        self.addCleanup(lambda: os.environ.clear(), original_env)

    def test_pipeline_processes_fixture(self):
        """Test that pipeline correctly processes fixture data"""

        # Copy fixture PDF to input directory
        fixture_pdf = Path("tests/fixtures/example.pdf")
        shutil.copy2(fixture_pdf, self.input_dir / fixture_pdf.name)

        # Load expected results
        expected_df = pd.read_csv(Path("tests/fixtures/expected.csv"), sep=';')
        expected_df['date'] = pd.to_datetime(expected_df['date'])

        # Run pipeline
        pipeline = create_default_pipeline(plot_labs=False)
        pipeline.execute()

        # Load results
        actual_df = pd.read_csv(self.output_dir / "merged_results.csv", sep=';')
        actual_df['date'] = pd.to_datetime(actual_df['date'])

        # Get detailed comparison
        differences = compare_dataframes(actual_df, expected_df)

        # Prepare error message if there are differences
        error_msg = []

        if differences['total_rows']['actual'] != differences['total_rows']['expected']:
            error_msg.append(f"\nRow count mismatch: actual={differences['total_rows']['actual']}, expected={differences['total_rows']['expected']}")

        if differences['missing_rows']:
            error_msg.append("\nMissing rows (in expected but not in actual):")
            for row in differences['missing_rows']:
                error_msg.append(f"  {row}")

        if differences['extra_rows']:
            error_msg.append("\nExtra rows (in actual but not in expected):")
            for row in differences['extra_rows']:
                error_msg.append(f"  {row}")

        if differences['matching_differences']:
            error_msg.append("\nDifferences in matching rows:")
            for diff in differences['matching_differences']:
                error_msg.append(f"  Lab: {diff['lab_name']}")
                for col, values in diff['differences'].items():
                    error_msg.append(f"    {col}: actual={values['actual']}, expected={values['expected']}")

        # Assert with detailed message
        assert not any([
            differences['missing_rows'],
            differences['extra_rows'],
            differences['matching_differences']
        ]), "\n".join(error_msg)

if __name__ == '__main__':
    unittest.main(verbosity=2)
