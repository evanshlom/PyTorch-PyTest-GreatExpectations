# Demo Instructions

VS Code → open folder → reopen in container when prompted. Then run these files in order:

## Python files to run:

### 1. `python train.py`
- Trains model, saves `model.pth` and `metrics.json`
- Shows training progress every 10 epochs

### 2. `pytest test_model.py -v`
- Runs 6 unit tests
- Shows all tests passing with green checkmarks

### 3. `python validation.py`
- Sets up Great Expectations validations
- Validates sample data
- Shows model performance health checks

**Note:** `model.py` and `data.py` are imported by other files - don't run directly.

## Expected output:
- Training loss decreasing
- All tests passing
- Data validation ✅
- Model health checks ✅