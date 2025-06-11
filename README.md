# PyTorch Stock Price Prediction

Minimal PyTorch regression model for predicting semiconductor stock prices.

# Demo Commands

## Production Workflow

### Step 1: Run Unit Tests
```bash
pytest test_model.py -v -s
```
Verify code quality first - all 8 tests should pass with descriptive output.

### Step 2: Validate Data Quality  
```bash
python validate_data.py
```
Check data quality BEFORE training:
- Schema validation (8 required columns)
- No null values allowed
- Value range checks (no negatives)
- Demo of bad data detection

### Step 3: Train Model
```bash
python train.py
```
Train only after confirming data quality. Shows Loss, MAE, and R² progress every 10 epochs.

### Step 4: Validate Model Performance
```bash
python validate_model.py
```
Check model performance AFTER training:
- Convergence: >20% loss reduction
- Overfitting: Val/Train ratio <1.5
- R² threshold: >0.1 (better than baseline)
- MAE acceptability: <$100

## Quick One-Liner
```bash
pytest test_model.py -v -s && python validate_data.py && python train.py && python validate_model.py
```

## Clean Start
```bash
rm -rf gx/ model.pth metrics.json
```

## Expected Output
- 8 unit tests pass with descriptions
- Good data passes all checks
- Bad data fails with detailed errors
- Model converges with positive R²
- All 4 performance criteria pass

That's the complete production workflow.