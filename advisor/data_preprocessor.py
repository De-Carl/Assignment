import pandas as pd
import numpy as np
from ast import literal_eval

class DataPreprocessor:
    """
    A class to load and preprocess the AI job market dataset.
    """
    def __init__(self, file_path='..\\dataset\\ai_job_market_unified.csv'):
        """
        Initializes the preprocessor with the path to the dataset.
        
        :param file_path: str, path to the CSV file.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Loads the dataset from the specified file path.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
            return self.df
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
            return None

    def _process_salary(self):
        """
        Processes the 'salary_range_usd' column to create an 'salary_avg_usd' column.
        It handles potential errors by splitting the string and calculating the mean.
        """
        if 'salary_range_usd' in self.df.columns:
            # Using a function to apply to each row
            def get_avg_salary(salary_range):
                if isinstance(salary_range, str):
                    try:
                        parts = salary_range.split('-')
                        if len(parts) == 2:
                            low, high = map(float, parts)
                            return (low + high) / 2
                    except (ValueError, TypeError):
                        # Handles cases where conversion fails
                        return np.nan
                return np.nan # Return NaN for non-string or malformed entries

            self.df['salary_avg_usd'] = self.df['salary_range_usd'].apply(get_avg_salary)
            
            # If the original dataset already contained an average, we can use it as a fallback
            # but recalculating ensures consistency.
            # We can fill any remaining NaNs with the median of the new column.
            if self.df['salary_avg_usd'].isnull().any():
                median_salary = self.df['salary_avg_usd'].median()
                self.df['salary_avg_usd'].fillna(median_salary, inplace=True)
            print("Salary processing complete.")
        else:
            print("Warning: 'salary_range_usd' column not found.")

    def _process_skills(self):
        """
        Processes the 'skills_required' column, converting string representation of a list
        into an actual list of skills.
        """
        if 'skills_required' in self.df.columns:
            # The 'skills_list' seems to be a pre-parsed version in the csv snippet
            # If 'skills_list' exists and is a stringified list, we use it. Otherwise, we parse 'skills_required'.
            target_col = 'skills_list' if 'skills_list' in self.df.columns else 'skills_required'
            
            # literal_eval is safer than eval for converting string representations of python objects
            self.df['skills_list_processed'] = self.df[target_col].apply(
                lambda x: literal_eval(x) if isinstance(x, str) and x.startswith('[') else x.split(',') if isinstance(x, str) else []
            )
            print("Skills processing complete.")
        else:
             print("Warning: 'skills_required' or 'skills_list' column not found.")
    
    def _process_dates(self):
        """
        Converts the 'posted_date' column to datetime objects.
        """
        if 'posted_date' in self.df.columns:
            self.df['posted_date'] = pd.to_datetime(self.df['posted_date'], errors='coerce')
            # errors='coerce' will turn unparseable dates into NaT (Not a Time)
            print("Date processing complete.")
        else:
            print("Warning: 'posted_date' column not found.")


    def run_preprocessing(self):
        """
        Runs the full preprocessing pipeline.
        """
        if self.load_data() is None:
            return None
        
        print("\n--- Starting Data Preprocessing ---")
        self._process_salary()
        self._process_skills()
        self._process_dates()
        print("--- Data Preprocessing Finished ---\n")
        return self.df

    def inspect_data(self):
        """
        Performs a basic inspection of the dataframe, checking for missing values and data types.
        """
        if self.df is None:
            print("Data not loaded. Please run preprocessing first.")
            return

        print("--- Data Inspection ---")
        print("Shape of the dataset (rows, columns):", self.df.shape)
        
        print("\nData Types:")
        print(self.df.info())

        print("\nMissing Values (Top 10):")
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
        print(missing_values.head(10))
        if missing_values.empty:
            print("No missing values found.")
        
        print("\n--- End of Inspection ---")


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.run_preprocessing()
    
    if processed_df is not None:
        preprocessor.inspect_data()
        
        print("\nSample of processed data:")
        # Display relevant new and old columns
        display_cols = [
            'job_title', 'salary_range_usd', 'salary_avg_usd', 
            'skills_required', 'skills_list_processed', 'posted_date'
        ]
        # Filter to columns that actually exist in the dataframe to avoid errors
        display_cols = [col for col in display_cols if col in processed_df.columns]
        print(processed_df[display_cols].head())
