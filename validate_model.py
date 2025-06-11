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
        return context
    
    # Create datasource for metrics
    datasource = context.sources.add_pandas("model_metrics_source")
    
    # Create expectation suite for model performance
    context.add_expectation_suite(suite_name)
    
    return context


def validate_model_performance(metrics_path='metrics.json'):
    """Validate model performance meets requirements"""
    
    if not os.path.exists(metrics_path):
        print("ERROR: No model found. Run train.py first!")
        return False
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Convert metrics to DataFrame for Great Expectations
    performance_data = {
        'metric': ['final_train_loss', 'final_val_loss', 'final_train_mae', 
                   'final_val_mae', 'final_train_r2', 'final_val_r2',
                   'loss_reduction', 'overfit_ratio'],
        'value': [
            metrics['train_loss'][-1],
            metrics['val_loss'][-1],
            metrics['train_mae'][-1],
            metrics['val_mae'][-1],
            metrics['train_r2'][-1],
            metrics['val_r2'][-1],
            metrics['train_loss'][-1] / metrics['train_loss'][0],  # loss reduction ratio
            metrics['val_loss'][-1] / metrics['train_loss'][-1]   # overfit ratio
        ]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Setup validation
    context = setup_model_validation()
    datasource = context.sources.add_or_update_pandas("metrics_validation")
    data_asset = datasource.add_dataframe_asset("model_metrics")
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Create validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="model_performance_suite"
    )
    
    # Define performance expectations
    print("Validating model performance criteria...")
    
    # 1. Model should converge (loss reduction > 20%)
    validator.expect_column_values_to_be_between(
        column='value',
        min_value=0,
        max_value=0.8,
        row_condition='metric=="loss_reduction"',
        result_format={'result_format': 'BOOLEAN_ONLY'}
    )
    
    # 2. Model shouldn't overfit too badly (val/train ratio < 1.5)
    validator.expect_column_values_to_be_between(
        column='value',
        min_value=0,
        max_value=1.5,
        row_condition='metric=="overfit_ratio"',
        result_format={'result_format': 'BOOLEAN_ONLY'}
    )
    
    # 3. R² should be positive (model better than baseline)
    validator.expect_column_values_to_be_between(
        column='value',
        min_value=0.1,
        row_condition='metric=="final_val_r2"',
        result_format={'result_format': 'BOOLEAN_ONLY'}
    )
    
    # 4. MAE should be reasonable (< $100)
    validator.expect_column_values_to_be_between(
        column='value',
        max_value=100,
        row_condition='metric=="final_val_mae"',
        result_format={'result_format': 'BOOLEAN_ONLY'}
    )
    
    # Save and run validation
    context.save_expectation_suite(
        expectation_suite=validator.get_expectation_suite(),
        expectation_suite_name="model_performance_suite"
    )
    
    results = validator.validate()
    
    # Display results
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS:")
    print("="*60)
    print(f"Training samples: 160")
    print(f"Validation samples: 40")
    print(f"Epochs trained: 50")
    print(f"Final learning achieved: {(1 - metrics['train_loss'][-1]/metrics['train_loss'][0])*100:.1f}% loss reduction")
    
    print("\n" + "-"*60)
    print("FINAL METRICS:")
    print("-"*60)
    print(f"Loss:  Train={metrics['train_loss'][-1]:>8.2f}  Val={metrics['val_loss'][-1]:>8.2f}")
    print(f"MAE:   Train=${metrics['train_mae'][-1]:>7.2f}  Val=${metrics['val_mae'][-1]:>7.2f}")
    print(f"R²:    Train={metrics['train_r2'][-1]:>8.4f}  Val={metrics['val_r2'][-1]:>8.4f}")
    
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION RESULTS:")
    print("="*60)
    
    # Check specific criteria with meaningful messages
    initial_loss = metrics['train_loss'][0]
    final_loss = metrics['train_loss'][-1]
    convergence_pct = (1 - final_loss/initial_loss) * 100
    
    checks = {
        "Model Converged": {
            "passed": final_loss < initial_loss * 0.8,
            "detail": f"{convergence_pct:.1f}% loss reduction (need >20%)"
        },
        "Not Overfitting": {
            "passed": metrics['val_loss'][-1] < metrics['train_loss'][-1] * 1.5,
            "detail": f"Val/Train ratio: {metrics['val_loss'][-1]/metrics['train_loss'][-1]:.2f} (need <1.5)"
        },
        "R² Positive": {
            "passed": metrics['val_r2'][-1] > 0.1,
            "detail": f"Val R²: {metrics['val_r2'][-1]:.3f} (need >0.1)"
        },
        "MAE Acceptable": {
            "passed": metrics['val_mae'][-1] < 100,
            "detail": f"Val MAE: ${metrics['val_mae'][-1]:.2f} (need <$100)"
        }
    }
    
    all_passed = True
    for check_name, check_data in checks.items():
        status = "PASS" if check_data["passed"] else "FAIL"
        print(f"[{status}] {check_name}: {check_data['detail']}")
        if not check_data["passed"]:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("MODEL PERFORMANCE VALIDATION")
    print("=" * 60)
    print("Purpose: Validate model performance AFTER training")
    print("Checks: Convergence, overfitting, R², MAE")
    print("=" * 60)
    print()
    
    passed = validate_model_performance()
    
    print("\n" + "="*60)
    if passed:
        print("RESULT: Model meets all performance criteria")
        print("Status: Ready for deployment")
    else:
        print("RESULT: Model needs improvement")
        print("\nRecommendations:")
        print("- Increase training epochs")
        print("- Tune hyperparameters") 
        print("- Add more training data")
        print("- Check feature engineering")