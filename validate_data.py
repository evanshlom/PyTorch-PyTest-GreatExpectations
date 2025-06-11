"""Data quality validation using Great Expectations - run BEFORE training"""
import great_expectations as gx
import pandas as pd
from data import create_sample_data
from data_bad import create_bad_sample_data


def setup_data_validation():
    """Setup Great Expectations for data quality checks"""
    context = gx.get_context()
    
    # Check if data validation suite exists
    suite_name = "data_quality_suite"
    existing_suites = context.list_expectation_suite_names()
    
    if suite_name in existing_suites:
        return context
    
    # Create datasource
    datasource = context.sources.add_pandas("data_validation_source")
    
    # Create sample data for setup
    df = create_sample_data(100)
    data_asset = datasource.add_dataframe_asset("training_data")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Create expectation suite for data quality
    context.add_expectation_suite(suite_name)
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )
    
    # Data quality expectations
    print("Setting up data quality expectations...")
    
    # 1. Schema validation - all columns must exist
    validator.expect_table_columns_to_match_ordered_list(
        column_list=['pe_ratio', 'dividend_yield', 'market_cap', 
                    'trading_volume', 'employee_count', 'profit_8k', 
                    'profit_10k', 'stock_price']
    )
    
    # 2. No null values allowed
    for column in df.columns:
        validator.expect_column_values_to_not_be_null(column=column)
    
    # 3. Data type validation
    validator.expect_column_values_to_be_between('pe_ratio', min_value=0, max_value=100)
    validator.expect_column_values_to_be_between('dividend_yield', min_value=0, max_value=10)
    validator.expect_column_values_to_be_between('market_cap', min_value=0)
    validator.expect_column_values_to_be_between('employee_count', min_value=1)
    validator.expect_column_values_to_be_between('stock_price', min_value=0)
    
    # Save suite
    context.save_expectation_suite(
        expectation_suite=validator.get_expectation_suite(),
        expectation_suite_name=suite_name
    )
    
    return context


def validate_training_data(df):
    """Validate data quality before training"""
    context = setup_data_validation()
    
    # Create validator
    datasource = context.sources.add_or_update_pandas("validation_source")
    data_asset = datasource.add_dataframe_asset("data_to_validate")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="data_quality_suite"
    )
    
    # Run validation
    results = validator.validate()
    
    return results


def show_validation_details(results):
    """Display detailed validation results"""
    if results.success:
        return
    
    print("\nDETAILED VALIDATION FAILURES:")
    print("-" * 60)
    
    for result in results.results:
        if not result.success:
            expectation_type = result.expectation_config.expectation_type
            
            if "column_values_to_be_between" in expectation_type:
                column = result.expectation_config.kwargs.get('column')
                min_val = result.expectation_config.kwargs.get('min_value', 'N/A')
                max_val = result.expectation_config.kwargs.get('max_value', 'N/A')
                unexpected = result.result.get('unexpected_count', 0)
                print(f"  [{column}] Out of range values: {unexpected} rows")
                print(f"    Expected: {min_val} to {max_val}")
                
            elif "column_values_to_not_be_null" in expectation_type:
                column = result.expectation_config.kwargs.get('column')
                unexpected = result.result.get('unexpected_count', 0)
                print(f"  [{column}] Null values found: {unexpected} rows")
                
            elif "columns_to_match_ordered_list" in expectation_type:
                print(f"  [SCHEMA] Column mismatch - missing required columns")


if __name__ == "__main__":
    print("DATA QUALITY VALIDATION")
    print("=" * 60)
    print("Purpose: Ensure data quality BEFORE training")
    print("Checks: Schema, nulls, value ranges")
    print("=" * 60)
    
    # Validate good training data
    print("\n[1] Validating training data quality...")
    train_df = create_sample_data(200)
    results = validate_training_data(train_df)
    
    if results.success:
        print("\nResult: PASSED")
        print(f"  - All {results.statistics['successful_expectations']} quality checks passed")
        print(f"  - {len(train_df)} rows validated")
        print(f"  - Ready for model training")
    else:
        print("\nResult: FAILED")
        print(f"  - Failed {results.statistics['unsuccessful_expectations']} checks")
        show_validation_details(results)
        print("\nAction: Fix data issues before training!")
        exit(1)
    
    # Demo what bad data looks like
    print("\n" + "-"*60)
    print("[2] Demo: What happens with bad data?")
    bad_df = create_bad_sample_data(50)
    bad_results = validate_training_data(bad_df)
    
    if not bad_results.success:
        print(f"\nResult: FAILED - {bad_results.statistics['unsuccessful_expectations']} issues found")
        show_validation_details(bad_results)
        print("\nThis is why we validate before training!")
    
    print("\n" + "="*60)
    print("CONCLUSION: Only train with quality-validated data")
    print("Next step: python train.py")