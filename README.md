# PyTorch Stock Price Prediction

Minimal PyTorch regression model for predicting semiconductor stock prices.

## Features
- **Input**: PE ratio, dividend yield, market cap, trading volume, employee count, profit (8K), profit (10K)
- **Output**: Stock price prediction

## Quick Start

### Using Dev Container (Recommended)
1. Open in VS Code
2. Install Remote-Containers extension
3. Reopen in Container

### Manual Setup
```bash
pip install -r requirements.txt
```

## Usage

### Train Model
```bash
python train.py
```

### Run Tests
```bash
pytest test_model.py -v
```

### Validate Data
```bash
python validation.py
```

## Project Structure
- `model.py` - Neural network definition
- `data.py` - Data handling utilities
- `train.py` - Training script
- `test_model.py` - Unit tests
- `validation.py` - Data validation & model health checks

## Model Architecture
- Input Layer: 7 features
- Hidden Layer 1: 32 neurons (ReLU)
- Hidden Layer 2: 16 neurons (ReLU)
- Output Layer: 1 neuron (price prediction)