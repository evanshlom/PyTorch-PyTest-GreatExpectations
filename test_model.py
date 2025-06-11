import pytest
import torch
import pandas as pd
import numpy as np
from model import StockPriceModel
from data import StockDataset, normalize_features, create_sample_data


def test_model_output_shape():
    """Test if model outputs correct shape for batch predictions"""
    model = StockPriceModel()
    # Create dummy input (batch_size=5, features=7)
    dummy_input = torch.randn(5, 7)
    output = model(dummy_input)
    
    assert output.shape == (5, 1), f"Expected shape (5, 1), got {output.shape}"
    print("Model output shape: Correct for batch size 5")


def test_model_forward_pass():
    """Test if model can process single sample without errors"""
    model = StockPriceModel()
    input_tensor = torch.randn(1, 7)
    
    # Should not raise any errors
    output = model(input_tensor)
    assert output is not None
    print("Model forward pass: Successfully processes single sample")


def test_dataset_length():
    """Test if dataset correctly reports its size"""
    features = torch.randn(10, 7)
    targets = torch.randn(10)
    dataset = StockDataset(features, targets)
    
    assert len(dataset) == 10
    print("Dataset length: Correctly reports 10 samples")


def test_dataset_getitem():
    """Test if dataset returns correct items by index"""
    features = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    targets = torch.tensor([100.0])
    dataset = StockDataset(features, targets)
    
    feature, target = dataset[0]
    assert torch.allclose(feature, features[0])
    assert torch.allclose(target, targets[0])
    print("Dataset indexing: Correctly retrieves samples")


def test_create_sample_data():
    """Test if sample data has correct structure and correlations"""
    df = create_sample_data(50)
    
    # Check shape
    assert df.shape == (50, 8)
    
    # Check columns
    expected_columns = ['pe_ratio', 'dividend_yield', 'market_cap', 
                       'trading_volume', 'employee_count', 'profit_8k', 
                       'profit_10k', 'stock_price']
    assert list(df.columns) == expected_columns
    
    # Check that stock price has some correlation with features
    # (since we added correlations in create_sample_data)
    pe_correlation = df['pe_ratio'].corr(df['stock_price'])
    assert pe_correlation > 0.3, "Stock price should correlate with PE ratio"
    print(f"Data generation: Created correlated data (r={pe_correlation:.2f})")


def test_normalize_features():
    """Test if feature normalization scales data correctly"""
    df = create_sample_data(20)
    features, targets = normalize_features(df)
    
    # Check shapes
    assert features.shape == (20, 7)
    assert targets.shape == (20,)
    
    # Check normalization (all features should be between 0 and 1)
    assert features.min() >= 0
    assert features.max() <= 1
    print("Feature normalization: All values scaled to [0,1] range")


def test_calculate_mae():
    """Test Mean Absolute Error calculation accuracy"""
    # Import here to avoid circular imports
    from train import calculate_mae
    
    predictions = torch.tensor([100.0, 150.0, 200.0])
    targets = torch.tensor([110.0, 140.0, 190.0])
    
    mae = calculate_mae(predictions, targets)
    expected_mae = 10.0  # Average of |10|, |10|, |10|
    
    assert abs(mae - expected_mae) < 0.01
    print(f"MAE calculation: Correctly computed as ${mae:.2f}")


def test_calculate_r2():
    """Test R-squared calculation for model fit quality"""
    # Import here to avoid circular imports
    from train import calculate_r2
    
    # Perfect prediction should give R² = 1
    targets = torch.tensor([100.0, 200.0, 300.0])
    perfect_predictions = targets.clone()
    
    r2_perfect = calculate_r2(perfect_predictions, targets)
    assert abs(r2_perfect - 1.0) < 0.01
    
    # Mean prediction should give R² = 0
    mean_predictions = torch.full_like(targets, targets.mean())
    r2_mean = calculate_r2(mean_predictions, targets)
    assert abs(r2_mean - 0.0) < 0.01
    print(f"R² calculation: Perfect={r2_perfect:.3f}, Baseline={r2_mean:.3f}")