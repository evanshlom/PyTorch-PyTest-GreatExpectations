"""Test Great Expectations with bad data"""
from validation import validate_data
from data_bad import create_bad_sample_data, create_missing_column_data, create_extreme_outlier_data

print("=" * 60)
print("Testing Great Expectations with Bad Data")
print("=" * 60)

# Test 1: Data with negative values and nulls
print("\n1. Testing data with negative values and nulls...")
bad_df = create_bad_sample_data(50)
print(f"   - Negative stock prices: {(bad_df['stock_price'] < 0).sum()}")
print(f"   - Negative PE ratios: {(bad_df['pe_ratio'] < 0).sum()}")
print(f"   - Null values: {bad_df.isnull().sum().sum()}")
results = validate_data(bad_df)

# Test 2: Data with missing columns
print("\n2. Testing data with missing columns...")
try:
    missing_df = create_missing_column_data(20)
    print(f"   - Columns: {list(missing_df.columns)}")
    results = validate_data(missing_df)
except Exception as e:
    print(f"   ❌ Failed as expected: {type(e).__name__}")

# Test 3: Data with extreme outliers
print("\n3. Testing data with extreme outliers...")
outlier_df = create_extreme_outlier_data(20)
print(f"   - Max PE ratio: {outlier_df['pe_ratio'].max()}")
print(f"   - Max stock price: ${outlier_df['stock_price'].max():,.2f}")
results = validate_data(outlier_df)

print("\n" + "=" * 60)
print("Bad data testing complete!")
print("This demonstrates how Great Expectations catches data quality issues.")
print("=" * 60)