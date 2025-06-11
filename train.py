import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import StockPriceModel
from data import create_sample_data, get_dataloaders
import json


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(predictions - targets)).item()


def calculate_r2(predictions, targets):
    """Calculate R-squared (coefficient of determination)"""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def train_model(model, train_loader, val_loader, epochs=50):
    """Train the model and track metrics"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features).squeeze()
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_preds_tensor = torch.tensor(train_predictions)
        train_targs_tensor = torch.tensor(train_targets)
        val_preds_tensor = torch.tensor(val_predictions)
        val_targs_tensor = torch.tensor(val_targets)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_mae = calculate_mae(train_preds_tensor, train_targs_tensor)
        val_mae = calculate_mae(val_preds_tensor, val_targs_tensor)
        train_r2 = calculate_r2(train_preds_tensor, train_targs_tensor)
        val_r2 = calculate_r2(val_preds_tensor, val_targs_tensor)
        
        # Track metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['train_mae'].append(train_mae)
        metrics['val_mae'].append(val_mae)
        metrics['train_r2'].append(train_r2)
        metrics['val_r2'].append(val_r2)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            print(f"           Train MAE = {train_mae:.2f}, Val MAE = {val_mae:.2f}")
            print(f"           Train R² = {train_r2:.4f}, Val R² = {val_r2:.4f}")
    
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