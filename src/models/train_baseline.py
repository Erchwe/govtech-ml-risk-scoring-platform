import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RiskScoringModel(nn.Module):
    """
    Simple Feedforward Neural Network for binary risk classification.
    Architecture: Input -> Dense(64) -> Dense(32) -> Output(1)[cite: 98, 99, 100, 101].
    """
    def __init__(self, input_dim):
        super(RiskScoringModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train_baseline():
    # Load dataset generated in the previous step
    df = pd.read_csv('src/data/claims_dataset.csv')
    X = df.drop('label', axis=1).values
    y = df['label'].values.reshape(-1, 1)

    # Split and scale data for training [cite: 195]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)

    # Initialize model, loss function, and optimizer
    model = RiskScoringModel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy Loss [cite: 103]
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam Optimizer [cite: 105]

    # Training loop execution
    print("Starting baseline model training...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}")

    # Save model state for future inference or containerization
    torch.save(model.state_dict(), 'src/models/baseline_model.pth')
    print("Baseline model saved to src/models/baseline_model.pth.")

if __name__ == "__main__":
    train_baseline()