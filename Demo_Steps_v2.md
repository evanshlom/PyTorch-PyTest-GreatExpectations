# Final Demo Commands - PyTorch Stock Prediction

## Pre-Demo Setup (Do This First!)

### 1. Verify Your Setup
```bash
python check_setup.py
```
Should show all files present and 8 tests found.

### 2. Fix Any Test Issues
```bash
python fix_and_run_tests.py
```
If tests fail, make sure `test_model.py` uses float tensors (1.0, not 1).

## Demo Commands (For YouTube)

### Option A: Automatic Demo (Recommended)
```bash
python demo_all.py
```
This runs everything in order with nice formatting.

### Option B: Manual Step-by-Step

#### 1. Train the Model
```bash
python train.py
```
**Shows:**
- Epoch progress every 10 epochs
- Loss, MAE (in dollars), and R² values
- Creates `model.pth` and `metrics.json`

#### 2. Run Unit Tests
```bash
pytest test_model.py -v
```
**Shows:**
- 8 tests all passing ✅
- Tests for model, data, and metrics

#### 3. Setup & Run Data Validation
```bash
# First time only - setup Great Expectations
python fix_validation.py

# Then run validation
python validation.py
```
**Shows:**
- ✅ Data validation passed
- Model performance health checks

#### 4. Demo Bad Data Detection
```bash
# Show what bad data looks like
python data_bad.py

# Test validation catches bad data
python test_bad_data.py
```
**Shows:**
- Different types of bad data
- ❌ Validation failures (this is good!)
- Specific issues caught

## Quick Commands for Demo

```bash
# Everything at once
python demo_all.py

# Or step by step
python train.py                # Train with metrics
pytest test_model.py -v        # Run 8 tests
python validation.py           # Validate good data
python test_bad_data.py        # Show bad data fails
```

## What to Emphasize in Video

1. **Model Metrics**: "Notice we track more than just loss - MAE tells us average price error in dollars, R² shows model fit"

2. **Testing**: "8 unit tests ensure our code works correctly"

3. **Data Validation**: "Great Expectations catches data issues before they break our model"

4. **Bad Data Demo**: "In real world, bad data is common - validation saves hours of debugging"

5. **Dev Container**: "Same environment for everyone - no 'works on my machine' issues"

## If Something Goes Wrong

```bash
# Reset validation
rm -rf gx/
python fix_validation.py

# Check what's wrong
python check_setup.py

# Verify tests work
python verify_tests.py
```