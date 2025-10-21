import pandas as pd
from pathlib import Path

class DataIntegrator:
    def __init__(self):
        self.data = None
        self.files_processed = 0
    
    def find_files(self, directory, pattern="*.csv"):
        directory = Path(directory)
        files = list(directory.glob(pattern))
        files.sort()
        print(f"Found {len(files)} files")
        return files
    
    def merge_files(self, file_paths, sample_size=None):
        all_data = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.stem
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                all_data.append(df)
                self.files_processed += 1
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"Combined data: {len(self.data)} rows")
        return self.data
