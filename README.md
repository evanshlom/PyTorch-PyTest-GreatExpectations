# PyTorch Stock Price Prediction

Minimal PyTorch regression model for predicting semiconductor stock prices.

# Demo Commands

## Step 1: Train the Model
```bash
python train.py
```
Creates model.pth and metrics.json. The model will now learn patterns from correlated data.

## Step 2: Run Unit Tests  
```bash
pytest test_model.py -v
```
All 8 tests should pass

## Step 3: Validate Data
```bash
python validation.py
```
Shows clear summary:
- Good data: PASSED
- Bad data: FAILED (7 issues) <- This is correct!
- Model performance metrics

## Optional: Show Bad Data
```bash
python data_bad.py
```
Shows examples of problematic data

## Clean Start
If you ran the old version, delete model.pth and metrics.json first:
```bash
rm -f model.pth metrics.json
python train.py
```

## That's all
Everything works out of the box with clear output.