import pandas as pd
import numpy as np

def generate_risk_dataset(n_rows=150000):
    """
    Generates a synthetic public service claim risk dataset for binary classification.
    
    Features included:
    - claim_amount: size of the claim [cite: 74]
    - claim_frequency_30d: count of claims in the last 30 days [cite: 75]
    - region_code: categorical encoding for regions [cite: 76]
    - service_type: categorical encoding for service categories [cite: 77]
    - time_since_last_claim: days since the previous claim [cite: 78]
    - historical_flag_rate: historical fraud rate for the user [cite: 79]
    - monthly_income_ratio: ratio of claim to monthly income [cite: 81]
    - anomaly_score_rule_based: heuristic anomaly score [cite: 80]
    """
    np.random.seed(42)
    
    # Generate base features with realistic distributions
    data = {
        'claim_amount': np.random.exponential(scale=500, size=n_rows),
        'claim_frequency_30d': np.random.poisson(lam=2, size=n_rows),
        'region_code': np.random.randint(0, 5, size=n_rows),
        'service_type': np.random.randint(0, 3, size=n_rows),
        'time_since_last_claim': np.random.uniform(0, 365, size=n_rows),
        'historical_flag_rate': np.random.beta(a=2, b=5, size=n_rows),
        'monthly_income_ratio': np.random.uniform(0.1, 0.9, size=n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate rule-based anomaly score as an additional feature [cite: 80]
    df['anomaly_score_rule_based'] = (df['claim_amount'] * df['claim_frequency_30d']) / 1000
    
    # Define fraud labels based on injected business rules 
    # Label 1 (High Risk) is triggered by high frequency and high amounts [cite: 72, 88]
    condition = (df['claim_amount'] > 1500) & (df['claim_frequency_30d'] > 5)
    df['label'] = condition.astype(int)
    
    # Inject 5% noise to simulate real-world data imperfections
    noise = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    df['label'] = np.where(noise == 1, 1 - df['label'], df['label'])
    
    return df

if __name__ == "__main__":
    # Generate and export the dataset to the local data directory
    dataset = generate_risk_dataset()
    dataset.to_csv('src/data/claims_dataset.csv', index=False)
    print(f"Dataset generated successfully with {len(dataset)} records.")