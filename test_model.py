import pytest
import torch
import pandas as pd
from model import StockPriceModel
from data import StockDataset, normalize_features, create_sample_data


def test_model_output_shape():
    """Test if model outputs correct shape"""
    model = StockPriceModel()
    # Create dummy input (batch_size=5, features=7)
    dummy_input = torch.randn(5, 7)
    output = model(dummy_input)
    
    assert output.shape == (5, 1), f"Expected shape (5, 1), got {output.shape}"


def test_model_forward_pass():
    """Test if model can do forward pass without errors"""
    model = StockPriceModel()
    input_tensor = torch.randn(1, 7)
    
    # Should not raise any errors
    output = model(input_tensor)
    assert output is not None


def test_dataset_length():
    """Test if dataset returns correct length"""
    features = torch.randn(10, 7)
    targets = torch.randn(10)
    dataset = StockDataset(features, targets)
    
    assert len(dataset) == 10


def test_dataset_getitem():
    """Test if dataset returns correct items"""
    features = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([100])
    dataset = StockDataset(features, targets)
    
    feature, target = dataset[0]
    assert torch.equal(feature, features[0])
    assert torch.equal(target, targets[0])


def test_create_sample_data():
    """Test if sample data has correct structure"""
    df = create_sample_data(50)
    
    # Check shape
    assert df.shape == (50, 8)
    
    # Check columns
    expected_columns = ['pe_ratio', 'dividend_yield', 'market_cap', 
                       'trading_volume', 'employee_count', 'profit_8k', 
                       'profit_10k', 'stock_price']
    assert list(df.columns) == expected_columns


def test_normalize_features():
    """Test feature normalization"""
    df = create_sample_data(20)
    features, targets = normalize_features(df)
    
    # Check shapes
    assert features.shape == (20, 7)
    assert targets.shape == (20,)
    
    # Check normalization (all features should be between 0 and 1)
    assert features.min() >= 0
    assert features.max() <= 1