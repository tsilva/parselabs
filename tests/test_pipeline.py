import unittest
from pathlib import Path
import pandas as pd
import tempfile
import shutil
import os
from main import Pipeline, PipelineConfig

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))
        
        # Save original env vars
        self.original_env = {}
        for key in ['SOURCE_PATH', 'DESTINATION_PATH']: self.original_env[key] = os.environ.get(key)
        
        # Setup temp paths
        self.temp_source = Path(self.temp_dir) / "source"
        self.temp_source.mkdir()

        self.temp_destination = Path(self.temp_dir) / "destination"
        self.temp_destination.mkdir()

        os.environ["SOURCE_PATH"] = str(self.temp_source)
        os.environ["DESTINATION_PATH"] = str(Path(self.temp_destination))
    
    def tearDown(self):
        """Restore environment after each test"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def prepare_test(self, test_pdf: Path, expected_csv: Path):
        """Copy test files to temporary location"""
        shutil.copy2(test_pdf, self.temp_source / test_pdf.name)
        self.expected_df = pd.read_csv(expected_csv, sep=';')
    
    def compare_results(self, actual_df: pd.DataFrame, expected_df: pd.DataFrame):
        """Compare pipeline output with expected results"""
        # Normalize dataframes
        for df in [actual_df, expected_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        sort_columns = ['date', 'lab_name', 'lab_value']
        actual_df = actual_df.sort_values(sort_columns).reset_index(drop=True)
        expected_df = expected_df.sort_values(sort_columns).reset_index(drop=True)
        
        # Compare dataframes
        pd.testing.assert_frame_equal(
            actual_df[sort_columns], 
            expected_df[sort_columns],
            check_dtype=False  # Allow string/object differences
        )
    
    def test_pipeline_output(self):
        """Test pipeline output matches expected results"""
        # Prepare test files
        test_pdf = Path("tests/fixtures/example.pdf")
        expected_csv = Path("tests/fixtures/expected.csv")
        self.prepare_test(test_pdf, expected_csv)
        
        # Run pipeline
        config = PipelineConfig(
            parallel_workers={
                "extract_pages": 1,  # Sequential for reproducibility
                "process_images": 1
            }
        )
        pipeline = Pipeline(config)
        pipeline.execute()
        
        # Load and compare results
        actual_df = pd.read_csv(Path(self.temp_dir) / "merged_results.csv", sep=';')
        self.compare_results(actual_df, self.expected_df)

if __name__ == '__main__':
    unittest.main(verbosity=2)
