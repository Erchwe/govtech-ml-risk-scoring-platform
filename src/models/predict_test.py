import torch
import numpy as np
from train_baseline import RiskScoringModel

def smoke_test_inference():
    """
    Loads the saved baseline model and performs a single inference
    to ensure the model artifact is functional.
    """
    # Define input dimension based on features (excluding label)
    # Features: claim_amount, freq_30d, region, service, time, hist_rate, income_ratio, anomaly_score
    input_dim = 8 
    
    # Initialize and load model
    model = RiskScoringModel(input_dim=input_dim)
    model.load_state_dict(torch.load('src/models/baseline_model.pth', weights_only=True))
    model.eval() # Set to evaluation mode
    
    # Create a dummy input (1 sample, 8 features)
    dummy_input = torch.randn(1, input_dim)
    
    with torch.no_grad():
        prediction = model(dummy_input)
        print(f"Test Prediction Output: {prediction.item():.4f}")
        
    if 0 <= prediction.item() <= 1:
        print("Smoke Test Passed: Model returned a valid probability.")
    else:
        print("Smoke Test Failed: Output is out of sigmoid range.")

if __name__ == "__main__":
    smoke_test_inference()