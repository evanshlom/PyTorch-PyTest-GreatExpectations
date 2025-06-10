import great_expectations as gx
import pandas as pd
from data import create_sample_data
import json


def create_expectations():
    """Set up Great Expectations for stock data validation"""
    context = gx.get_context()
    
    # Create datasource
    datasource = context.sources.add_pandas("stock_datasource")
    
    # Load data
    df = create_sample_data(100)
    data_asset = datasource.add_dataframe_asset("stock_data")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Create expectation suite
    context.add_expectation_suite("stock_data_suite")
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="stock_data_suite"
    )
    
    # Add expectations
    # 1. Check all required columns exist
    validator.expect_table_columns_to_match_ordered_list(
        column_list=['pe_ratio', 'dividend_yield', 'market_cap', 
                    'trading_volume', 'employee_count', 'profit_8k', 
                    'profit_10k', 'stock_price']
    )
    
    # 2. PE ratio should be positive
    validator.expect_column_values_to_be_between(
        column='pe_ratio', min_value=0, max_value=100
    )
    
    # 3. Dividend yield should be non-negative
    validator.expect_column_values_to_be_between(
        column='dividend_yield', min_value=0, max_value=10
    )
    
    # 4. Market cap should be positive
    validator.expect_column_values_to_be_between(
        column='market_cap', min_value=0
    )
    
    # 5. Employee count should be positive integers
    validator.expect_column_values_to_be_between(
        column='employee_count', min_value=1
    )
    
    # 6. Stock price should be positive
    validator.expect_column_values_to_be_between(
        column='stock_price', min_value=0
    )
    
    # 7. No null values allowed
    for column in df.columns:
        validator.expect_column_values_to_not_be_null(column=column)
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    return validator


def validate_data(df):
    """Run validation on new data"""
    context = gx.get_context()
    
    # Create checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name="stock_data_checkpoint",
        validations=[{
            "batch_request": {
                "datasource_name": "stock_datasource",
                "data_asset_name": "stock_data",
                "options": {"dataframe": df}
            },
            "expectation_suite_name": "stock_data_suite"
        }]
    )
    
    # Run validation
    results = checkpoint.run()
    
    # Check if validation passed
    if results.success:
        print("Data validation passed!")
    else:
        print("Data validation failed!")
        
    return results


def check_model_performance(metrics_path='metrics.json'):
    """Validate model performance metrics"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Performance checks
    final_train_loss = metrics['train_loss'][-1]
    final_val_loss = metrics['val_loss'][-1]
    
    checks = {
        'final_train_loss_reasonable': final_train_loss < 1000,
        'final_val_loss_reasonable': final_val_loss < 1000,
        'not_overfitting': final_val_loss < final_train_loss * 2,
        'model_converged': metrics['train_loss'][-1] < metrics['train_loss'][0] * 0.5
    }
    
    print("\n📊 Model Performance Health Check:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {passed}")
    
    return all(checks.values())


if __name__ == "__main__":
    # Create expectations
    validator = create_expectations()
    print("Expectations created!")
    
    # Validate sample data
    test_df = create_sample_data(50)
    results = validate_data(test_df)
    
    # Check model performance (if metrics exist)
    try:
        check_model_performance()
    except FileNotFoundError:
        print("No metrics.json found - run train.py first")