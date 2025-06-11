import pandas as pd
import numpy as np


def create_bad_sample_data(n_samples=100):
    """Generate problematic stock data to trigger validation failures"""
    np.random.seed(42)
    
    # Create data with various issues
    data = {
        'pe_ratio': np.concatenate([
            np.random.uniform(-10, 5, 20),  # Negative PE ratios (bad!)
            np.random.uniform(10, 50, n_samples-20)
        ]),
        'dividend_yield': np.concatenate([
            np.random.uniform(-2, 0, 15),  # Negative yields (bad!)
            np.random.uniform(0, 5, n_samples-15)
        ]),
        'market_cap': np.concatenate([
            np.random.uniform(-1e9, 0, 10),  # Negative market cap (bad!)
            np.random.uniform(1e9, 1e11, n_samples-10)
        ]),
        'trading_volume': np.random.uniform(1e6, 1e8, n_samples),
        'employee_count': np.concatenate([
            np.array([0, -100, -50]),  # Zero/negative employees (bad!)
            np.random.randint(1000, 50000, n_samples-3)
        ]),
        'profit_8k': np.random.uniform(-1e8, 5e8, n_samples),
        'profit_10k': np.random.uniform(-1e8, 5e8, n_samples),
        'stock_price': np.concatenate([
            np.random.uniform(-50, 0, 25),  # Negative prices (bad!)
            np.random.uniform(50, 500, n_samples-25)
        ])
    }
    
    df = pd.DataFrame(data)
    
    # Add some null values
    df.loc[5:10, 'pe_ratio'] = np.nan
    df.loc[15:18, 'dividend_yield'] = np.nan
    
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
    data = {
        'pe_ratio': np.concatenate([
            np.array([500, 1000, -200]),  # Extreme PE ratios
            np.random.uniform(10, 50, n_samples-3)
        ]),
        'dividend_yield': np.concatenate([
            np.array([50, 100]),  # Impossible dividend yields
            np.random.uniform(0, 5, n_samples-2)
        ]),
        'market_cap': np.random.uniform(1e9, 1e11, n_samples),
        'trading_volume': np.random.uniform(1e6, 1e8, n_samples),
        'employee_count': np.random.randint(1000, 50000, n_samples),
        'profit_8k': np.random.uniform(-1e8, 5e8, n_samples),
        'profit_10k': np.random.uniform(-1e8, 5e8, n_samples),
        'stock_price': np.concatenate([
            np.array([0, 10000, 50000]),  # Extreme prices
            np.random.uniform(50, 500, n_samples-3)
        ])
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Creating bad data samples...")
    
    # Show different types of bad data
    bad_df = create_bad_sample_data(50)
    print("\n1. Data with negative values and nulls:")
    print(bad_df.describe())
    print(f"\nNull values: \n{bad_df.isnull().sum()}")
    
    missing_df = create_missing_column_data(20)
    print("\n2. Data with missing columns:")
    print(f"Columns: {list(missing_df.columns)}")
    
    outlier_df = create_extreme_outlier_data(20)
    print("\n3. Data with extreme outliers:")
    print(outlier_df[['pe_ratio', 'dividend_yield', 'stock_price']].describe())