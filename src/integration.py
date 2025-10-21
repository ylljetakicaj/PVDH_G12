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

    def aggregate_by_time(self, date_col, freq='M'):
        if self.data is None:
            return None
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if freq == 'M':
            df['period'] = df[date_col].dt.to_period('M')
        elif freq == 'Y':
            df['period'] = df[date_col].dt.to_period('Y')
        elif freq == 'D':
            df['period'] = df[date_col].dt.date
        time_agg = df.groupby('period', observed=True).size().reset_index(name='count')
        return time_agg
    
    def aggregate_by_category(self, category_col):
        if self.data is None:
            return None
        cat_agg = self.data.groupby(category_col, observed=True).size().reset_index(name='count')
        return cat_agg.sort_values('count', ascending=False)
    
    def get_summary_stats(self):
        if self.data is None:
            return None
        stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'files_processed': self.files_processed,
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }
        return stats
    
    def save_data(self, output_path):
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
            return True
        return False
