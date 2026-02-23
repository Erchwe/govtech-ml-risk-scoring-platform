import pandas as pd

def verify_dataset(file_path):
    """
    Verifies the integrity of the generated synthetic dataset.
    Checks for row count, column existence, and label distribution.
    """
    df = pd.read_csv(file_path)
    
    # Shape
    rows, cols = df.shape
    print(f"Total Rows: {rows}")  
    print(f"Total Columns: {cols}")
    
    # Fraud Ratio
    fraud_ratio = (df['label'] == 1).mean() * 100
    print(f"Fraud (High Risk) Ratio: {fraud_ratio:.2f}%") 
    
    # Missing Values
    if df.isnull().values.any():
        print("Warning: Missing values detected!")
    else:
        print("Data Integrity: No missing values found.")

if __name__ == "__main__":
    verify_dataset('src/data/claims_dataset.csv')