import pandas as pd
import numpy as np


def create_bad_sample_data(n_samples=100):
    """Generate problematic stock data to trigger validation failures"""
    np.random.seed(42)
    
    # Ensure we have enough samples for negative values
    n_negative = min(25, n_samples // 4)
    n_negative_pe = min(20, n_samples // 5)
    n_negative_yield = min(15, n_samples // 6)
    n_negative_cap = min(10, n_samples // 10)
    
    # Create data with various issues
    data = {
        'pe_ratio': np.concatenate([
            np.random.uniform(-10, 5, n_negative_pe),  # Negative PE ratios
            np.random.uniform(10, 50, n_samples - n_negative_pe)
        ]),
        'dividend_yield': np.concatenate([
            np.random.uniform(-2, 0, n_negative_yield),  # Negative yields
            np.random.uniform(0, 5, n_samples - n_negative_yield)
        ]),
        'market_cap': np.concatenate([
            np.random.uniform(-1e9, 0, n_negative_cap),  # Negative market cap
            np.random.uniform(1e9, 1e11, n_samples - n_negative_cap)
        ]),
        'trading_volume': np.random.uniform(1e6, 1e8, n_samples),
        'employee_count': np.concatenate([
            np.array([0, -100, -50])[:min(3, n_samples)],  # Bad employee counts
            np.random.randint(1000, 50000, max(0, n_samples - 3))
        ]),
        'profit_8k': np.random.uniform(-1e8, 5e8, n_samples),
        'profit_10k': np.random.uniform(-1e8, 5e8, n_samples),
        'stock_price': np.concatenate([
            np.random.uniform(-50, 0, n_negative),  # Negative prices
            np.random.uniform(50, 500, n_samples - n_negative)
        ])
    }
    
    df = pd.DataFrame(data)
    
    # Add some null values
    if n_samples > 10:
        df.loc[5:10, 'pe_ratio'] = np.nan
        df.loc[15:min(18, n_samples-1), 'dividend_yield'] = np.nan
    
    return df


def create_missing_column_data(n_samples=50):
    """Create data missing required columns"""
    data = {
        'pe_ratio': np.random.uniform(10, 50, n_samples),
        'dividend_yield': np.random.uniform(0, 5, n_samples),
        # Missing: market_cap, trading_volume
        'employee_count': np.random.randint(1000, 50000, n_samples),
        'profit_8k': np.random.uniform(-1e8, 5e8, n_samples),
        'profit_10k': np.random.uniform(-1e8, 5e8, n_samples),
        'stock_price': np.random.uniform(50, 500, n_samples)
    }
    
    return pd.DataFrame(data)


def create_extreme_outlier_data(n_samples=50):
    """Create data with extreme outliers"""
    n_outliers = min(3, n_samples // 10)
    
    data = {
        'pe_ratio': np.concatenate([
            np.array([500, 1000, -200])[:n_outliers],  # Extreme PE ratios
            np.random.uniform(10, 50, n_samples - n_outliers)
        ]),
        'dividend_yield': np.concatenate([
            np.array([50, 100])[:min(2, n_outliers)],  # Impossible yields
            np.random.uniform(0, 5, n_samples - min(2, n_outliers))
        ]),
        'market_cap': np.random.uniform(1e9, 1e11, n_samples),
        'trading_volume': np.random.uniform(1e6, 1e8, n_samples),
        'employee_count': np.random.randint(1000, 50000, n_samples),
        'profit_8k': np.random.uniform(-1e8, 5e8, n_samples),
        'profit_10k': np.random.uniform(-1e8, 5e8, n_samples),
        'stock_price': np.concatenate([
            np.array([0, 10000, 50000])[:n_outliers],  # Extreme prices
            np.random.uniform(50, 500, n_samples - n_outliers)
        ])
    }
    
    return pd.DataFrame(data)