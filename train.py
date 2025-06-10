import torch
import torch.nn as nn
import torch.optim as optim
from model import StockPriceModel
from data import create_sample_data, get_dataloaders
import json


def train_model(model, train_loader, val_loader, epochs=50):
    """Train the model and track metrics"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features).squeeze()
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        # Track metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return metrics


def save_model_and_metrics(model, metrics):
    """Save trained model and performance metrics"""
    torch.save(model.state_dict(), 'model.pth')
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Model and metrics saved!")


if __name__ == "__main__":
    # Generate data
    df = create_sample_data(200)
    train_loader, val_loader = get_dataloaders(df)
    
    # Create and train model
    model = StockPriceModel()
    metrics = train_model(model, train_loader, val_loader)
    
    # Save everything
    save_model_and_metrics(model, metrics)