"""Model performance validation using Great Expectations - run AFTER training"""
import great_expectations as gx
import pandas as pd
import json
import os


def setup_model_validation():
    """Setup Great Expectations for model metrics validation"""
    context = gx.get_context()
    
    # Check if model validation suite exists
    suite_name = "model_performance_suite"
    existing_suites = context.list_expectation_suite_names()
    
    if suite_name in existing_suites:
        # Remove old suite to create fresh one
        context.delete_expectation_suite(suite_name)
    
    # Create datasource for metrics
    datasource = context.sources.add_or_update_pandas("model_metrics_source")
    
    # Create expectation suite for model performance
    context.add_expectation_suite(suite_name)
    
    return context, datasource


def validate_model_performance(metrics_path='metrics.json'):
    """Validate model performance meets requirements using Great Expectations"""
    
    if not os.path.exists(metrics_path):
        print("ERROR: No model found. Run train.py first!")
        return False
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print("MODEL PERFORMANCE VALIDATION")
    print("=" * 60)
    print("Purpose: Validate model performance AFTER training")
    print("Using: Great Expectations for metrics validation")
    print("=" * 60)
    print()
    
    # Convert metrics to DataFrame for Great Expectations
    # Create separate DataFrames for different types of checks
    
    # 1. Convergence check data
    convergence_df = pd.DataFrame({
        'loss_reduction_ratio': [metrics['train_loss'][-1] / metrics['train_loss'][0]]
    })
    
    # 2. Overfitting check data
    overfitting_df = pd.DataFrame({
        'validation_train_ratio': [metrics['val_loss'][-1] / metrics['train_loss'][-1]]
    })
    
    # 3. R² check data
    r2_df = pd.DataFrame({
        'validation_r2': [metrics['val_r2'][-1]]
    })
    
    # 4. MAE check data
    mae_df = pd.DataFrame({
        'validation_mae': [metrics['val_mae'][-1]]
    })
    
    # Setup validation
    context, datasource = setup_model_validation()
    
    # Create validators for each metric type
    results = {}
    all_passed = True
    
    print("Running Great Expectations validations...")
    print("-" * 60)
    
    # 1. Validate convergence
    print("\n[1] Checking Model Convergence...")
    conv_asset = datasource.add_dataframe_asset("convergence_metrics")
    conv_batch = conv_asset.build_batch_request(dataframe=convergence_df)
    conv_validator = context.get_validator(
        batch_request=conv_batch,
        expectation_suite_name="model_performance_suite"
    )
    
    conv_validator.expect_column_values_to_be_between(
        column='loss_reduction_ratio',
        max_value=0.8,  # Must reduce loss by at least 20%
        meta={"description": "Model should reduce loss by at least 20%"}
    )
    
    conv_results = conv_validator.validate()
    convergence_passed = conv_results.success
    results['convergence'] = convergence_passed
    
    # 2. Validate overfitting
    print("\n[2] Checking Overfitting...")
    overfit_asset = datasource.add_dataframe_asset("overfitting_metrics")
    overfit_batch = overfit_asset.build_batch_request(dataframe=overfitting_df)
    overfit_validator = context.get_validator(
        batch_request=overfit_batch,
        expectation_suite_name="model_performance_suite"
    )
    
    overfit_validator.expect_column_values_to_be_between(
        column='validation_train_ratio',
        max_value=1.5,  # Validation loss should not be >1.5x training loss
        meta={"description": "Validation/Training ratio should be < 1.5"}
    )
    
    overfit_results = overfit_validator.validate()
    overfitting_passed = overfit_results.success
    results['overfitting'] = overfitting_passed
    
    # 3. Validate R²
    print("\n[3] Checking R² Score...")
    r2_asset = datasource.add_dataframe_asset("r2_metrics")
    r2_batch = r2_asset.build_batch_request(dataframe=r2_df)
    r2_validator = context.get_validator(
        batch_request=r2_batch,
        expectation_suite_name="model_performance_suite"
    )
    
    r2_validator.expect_column_values_to_be_between(
        column='validation_r2',
        min_value=0.5,  # R² must be > 0.5
        meta={"description": "R² should be > 0.5 (strong correlation)"}
    )
    
    r2_results = r2_validator.validate()
    r2_passed = r2_results.success
    results['r2'] = r2_passed
    
    # 4. Validate MAE
    print("\n[4] Checking Mean Absolute Error...")
    mae_asset = datasource.add_dataframe_asset("mae_metrics")
    mae_batch = mae_asset.build_batch_request(dataframe=mae_df)
    mae_validator = context.get_validator(
        batch_request=mae_batch,
        expectation_suite_name="model_performance_suite"
    )
    
    mae_validator.expect_column_values_to_be_between(
        column='validation_mae',
        max_value=50,  # MAE must be < $50
        meta={"description": "MAE should be < $50"}
    )
    
    mae_results = mae_validator.validate()
    mae_passed = mae_results.success
    results['mae'] = mae_passed
    
    # Display detailed results
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS:")
    print("="*60)
    print(f"Training samples: 160")
    print(f"Validation samples: 40")
    print(f"Epochs trained: 100")
    print(f"Final learning achieved: {(1 - metrics['train_loss'][-1]/metrics['train_loss'][0])*100:.1f}% loss reduction")
    
    print("\n" + "-"*60)
    print("FINAL METRICS:")
    print("-"*60)
    print(f"Loss:  Train={metrics['train_loss'][-1]:>8.2f}  Val={metrics['val_loss'][-1]:>8.2f}")
    print(f"MAE:   Train=${metrics['train_mae'][-1]:>7.2f}  Val=${metrics['val_mae'][-1]:>7.2f}")
    print(f"R²:    Train={metrics['train_r2'][-1]:>8.4f}  Val={metrics['val_r2'][-1]:>8.4f}")
    
    print("\n" + "="*60)
    print("GREAT EXPECTATIONS VALIDATION RESULTS:")
    print("="*60)
    
    # Check results
    convergence_pct = (1 - metrics['train_loss'][-1]/metrics['train_loss'][0]) * 100
    val_train_ratio = metrics['val_loss'][-1]/metrics['train_loss'][-1]
    
    print(f"[{'PASS' if results['convergence'] else 'FAIL'}] Convergence Check: {convergence_pct:.1f}% reduction (need >20%)")
    print(f"[{'PASS' if results['overfitting'] else 'FAIL'}] Overfitting Check: {val_train_ratio:.2f} ratio (need <1.5)")
    print(f"[{'PASS' if results['r2'] else 'FAIL'}] R² Check: {metrics['val_r2'][-1]:.3f} (need >0.5)")
    print(f"[{'PASS' if results['mae'] else 'FAIL'}] MAE Check: ${metrics['val_mae'][-1]:.2f} (need <$50)")
    
    all_passed = all(results.values())
    
    # Save the complete suite
    context.save_expectation_suite(
        expectation_suite=conv_validator.get_expectation_suite(),
        expectation_suite_name="model_performance_suite"
    )
    
    print("\n" + "="*60)
    if all_passed:
        print("RESULT: Model meets all performance criteria")
        print("Status: Ready for deployment")
        print(f"Great Expectations validation: {len(results)} checks passed")
    else:
        print("RESULT: Model needs improvement")
        failed_checks = [k for k, v in results.items() if not v]
        print(f"Failed checks: {', '.join(failed_checks)}")
        print("\nRecommendations:")
        if not results.get('convergence'):
            print("- Train for more epochs")
        if not results.get('overfitting'):
            print("- Add regularization or dropout")
        if not results.get('r2'):
            print("- Check feature engineering")
        if not results.get('mae'):
            print("- Tune hyperparameters")
    
    return all_passed


if __name__ == "__main__":
    validate_model_performance()