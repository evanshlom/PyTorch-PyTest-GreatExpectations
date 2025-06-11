# PyTorch Stock Price Prediction

PyTorch model training with PyTest unit testing, and Great Expectations schema checks and metrics validation. **See "Production Workflow" section below for simple quick instructions.**

## Files
- `model.py` - Neural network definition
- `data.py` - Dataset and data utilities  
- `data_bad.py` - Generate problematic data for testing
- `train.py` - Training script with metrics
- `test_model.py` - Unit tests
- `validate_data.py` - Data quality validation (pre-training)
- `validate_model.py` - Model performance validation (post-training)

## Setup
1. Open in VS Code
2. Reopen in Dev Container when prompted
3. Container will build with all dependencies

## Production Workflow

### Step 1: Run Unit Tests
```bash
pytest test_model.py -v
```
Verifies code quality - all 8 tests should pass.

### Step 2: Validate Data Quality
```bash
python validate_data.py
```
Checks data quality BEFORE training:
- Schema validation
- No null values
- Value ranges
- Shows what bad data looks like

### Step 3: Train Model
```bash
python train.py
```
Trains model on validated data. Shows Loss, MAE, and R² metrics.

### Step 4: Validate Model Performance
```bash
python validate_model.py
```
Checks model performance AFTER training:
- Convergence criteria
- Overfitting detection
- R² threshold
- MAE acceptability

## Clean Start
```bash
rm -rf gx/ model.pth metrics.json
```

## Expected Results
- Tests: All 8 pass with descriptive output
- Data validation: Good data passes, bad data fails with details
- Training: R² > 0.7 (model learns simple PE-price relationship)
- Model validation: 2 pass (convergence, overfitting), 2 may fail (R², MAE)

Simple, clean, production-ready workflow.