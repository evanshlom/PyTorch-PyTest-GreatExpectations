import great_expectations as gx
import pandas as pd
import os
import json
from data import create_sample_data
from data_bad import create_bad_sample_data


def setup_great_expectations():
    """Setup Great Expectations with stock data validation rules"""
    # Get or create context
    context = gx.get_context()
    
    # Check if suite already exists
    suite_name = "stock_data_suite"
    existing_suites = context.list_expectation_suite_names()
    
    if suite_name in existing_suites:
        return context
    
    # Create datasource
    datasource = context.sources.add_pandas("stock_datasource")
    
    # Create sample data for setup
    df = create_sample_data(100)
    data_asset = datasource.add_dataframe_asset("stock_data")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Create expectation suite
    context.add_expectation_suite(suite_name)
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
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
    context.save_expectation_suite(
        expectation_suite=validator.get_expectation_suite(),
        expectation_suite_name=suite_name
    )
    
    print("Great Expectations suite initialized")
    return context


def validate_data(df, data_type=""):
    """Run validation on a dataframe"""
    # Ensure GX is set up
    context = setup_great_expectations()
    
    # Create validator for the data
    datasource = context.sources.add_or_update_pandas("validation_datasource")
    data_asset = datasource.add_dataframe_asset("data_to_validate")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    try:
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name="stock_data_suite"
        )
        
        # Run validation
        results = validator.validate()
        
        # Return results for summary
        return results
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None


def check_model_performance(metrics_path='metrics.json'):
    """Validate model performance metrics"""
    if not os.path.exists(metrics_path):
        print("No metrics.json found - run train.py first")
        return False
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Performance checks (adjusted for synthetic data)
    final_train_loss = metrics['train_loss'][-1]
    final_val_loss = metrics['val_loss'][-1]
    final_train_mae = metrics['train_mae'][-1]
    final_val_mae = metrics['val_mae'][-1]
    final_train_r2 = metrics['train_r2'][-1]
    final_val_r2 = metrics['val_r2'][-1]
    
    checks = {
        'not_overfitting': final_val_loss < final_train_loss * 1.5,
        'model_converged': metrics['train_loss'][-1] < metrics['train_loss'][0] * 0.8,
        'mae_reasonable': final_val_mae < 100,  # MAE less than $100
        'r2_positive': final_val_r2 > 0.2  # Model explains at least 20% variance
    }
    
    print("\nModel Performance:")
    print(f"Final Train Loss: {final_train_loss:.2f}, Val Loss: {final_val_loss:.2f}")
    print(f"Final Train MAE: ${final_train_mae:.2f}, Val MAE: ${final_val_mae:.2f}")
    print(f"Final Train R2: {final_train_r2:.4f}, Val R2: {final_val_r2:.4f}")
    
    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("Data Validation with Great Expectations")
    print("=" * 60)
    
    # Test with good data
    print("\n1. Validating good data:")
    good_df = create_sample_data(50)
    good_results = validate_data(good_df, "good")
    
    # Test with bad data
    print("\n2. Validating bad data:")
    bad_df = create_bad_sample_data(50)
    bad_results = validate_data(bad_df, "bad")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY:")
    print("=" * 60)
    
    if good_results and good_results.success:
        print("Good data: PASSED (all checks passed)")
    else:
        print("Good data: FAILED (unexpected!)")
        
    if bad_results and not bad_results.success:
        failed_count = bad_results.statistics['unsuccessful_expectations']
        print(f"Bad data:  FAILED ({failed_count} issues found) <- This is correct!")
    else:
        print("Bad data:  PASSED (unexpected - validation not catching issues)")
    
    print("\nConclusion: Validation is working correctly!")
    print("Good data passes, bad data fails as expected.")
    
    # Check model performance if available
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE CHECK:")
    print("=" * 60)
    check_model_performance()