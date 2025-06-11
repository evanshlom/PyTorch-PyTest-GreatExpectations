# PyTorch Stock Price Prediction

Minimal PyTorch regression model for predicting semiconductor stock prices.

# Demo Commands

## Production-Style Flow

### Step 1: Run Unit Tests
```bash
pytest test_model.py -v
```
Verify code quality first - all 8 tests should pass.

### Step 2: Validate Data Quality  
```bash
python validation.py
```
Check data quality before training:
- Good data: PASSED
- Bad data: FAILED (7 issues) <- This is correct!

### Step 3: Train Model
```bash
python train.py
```
Train only after confirming data quality. Shows Loss, MAE, and R² progress.

### Step 4: Final Validation
```bash
python validation.py
```
Now includes model performance metrics with the data validation.

## Quick One-Liner
```bash
pytest test_model.py -v && python validation.py && python train.py && python validation.py
```

## Clean Start
```bash
rm -rf gx/ model.pth metrics.json
```

## Optional: Examine Bad Data
```bash
python data_bad.py
```

That's the complete workflow - test code, validate data, train model, verify results.