import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """PyTorch dataset for stock data"""
    
    def __init__(self, features, targets):
        # Convert to float tensors, handling different input types
        if isinstance(features, torch.Tensor):
            self.features = features.float()
        else:
            self.features = torch.FloatTensor(features)
            
        if isinstance(targets, torch.Tensor):
            self.targets = targets.float()
        else:
            self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_sample_data(n_samples=100):
    """Generate synthetic stock data with correlations for demo"""
    np.random.seed(42)
    
    # Create base features with some correlations
    # Higher PE ratio tends to mean higher growth stocks = higher price
    pe_ratio = np.random.uniform(10, 50, n_samples)
    
    # Market cap strongly correlates with stock price
    market_cap = np.random.uniform(1e9, 1e11, n_samples)
    
    # Create other features
    dividend_yield = np.random.uniform(0, 5, n_samples)
    trading_volume = np.random.uniform(1e6, 1e8, n_samples)
    employee_count = np.random.randint(1000, 50000, n_samples)
    profit_8k = np.random.uniform(-1e8, 5e8, n_samples)
    profit_10k = np.random.uniform(-1e8, 5e8, n_samples)
    
    # Create stock price with correlations
    # Price = base + pe_contribution + market_cap_contribution + profit_contribution + noise
    base_price = 100
    pe_contribution = pe_ratio * 3  # Higher PE = higher price
    market_cap_contribution = (market_cap / 1e9) * 2  # Bigger company = higher price
    profit_contribution = (profit_10k / 1e8) * 10  # More profit = higher price
    noise = np.random.normal(0, 20, n_samples)  # Some randomness
    
    stock_price = base_price + pe_contribution + market_cap_contribution + profit_contribution + noise
    # Ensure positive prices
    stock_price = np.maximum(stock_price, 10)
    
    data = {
        'pe_ratio': pe_ratio,
        'dividend_yield': dividend_yield,
        'market_cap': market_cap,
        'trading_volume': trading_volume,
        'employee_count': employee_count,
        'profit_8k': profit_8k,
        'profit_10k': profit_10k,
        'stock_price': stock_price
    }
    
    return pd.DataFrame(data)


def normalize_features(df):
    """Simple normalization"""
    features = df.drop('stock_price', axis=1)
    targets = df['stock_price']
    
    # Normalize features (simple min-max)
    features_norm = (features - features.min()) / (features.max() - features.min())
    
    return features_norm.values, targets.values


def get_dataloaders(df, batch_size=32, train_split=0.8):
    """Create train/val dataloaders"""
    features, targets = normalize_features(df)
    
    # Split data
    split_idx = int(len(features) * train_split)
    
    train_dataset = StockDataset(features[:split_idx], targets[:split_idx])
    val_dataset = StockDataset(features[split_idx:], targets[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader