# One command runs everything!
python demo_all.py


python train.py                # Shows Loss, MAE, R²
pytest test_model.py -v        # 8 tests pass ✅
python validation.py           # Good data passes ✅
python test_bad_data.py        # Bad data fails ❌ (good!)