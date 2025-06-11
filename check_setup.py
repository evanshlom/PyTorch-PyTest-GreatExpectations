"""Check if all files are present and correct"""
import os
import importlib.util

print("🔍 Checking PyTorch Stock Prediction Project Setup")
print("=" * 60)

# Files to check
required_files = {
    "Core Files": [
        "model.py",
        "data.py", 
        "train.py",
        "test_model.py",
        "validation.py",
        "requirements.txt",
        "Dockerfile",
        "README.md"
    ],
    "Bad Data Testing": [
        "data_bad.py",
        "test_bad_data.py",
        "fix_validation.py"
    ],
    "Helper Scripts": [
        "demo_all.py",
        "verify_tests.py",
        "check_setup.py"
    ],
    "Dev Container": [
        ".devcontainer/devcontainer.json"
    ]
}

# Check files exist
missing_files = []
for category, files in required_files.items():
    print(f"\n{category}:")
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            missing_files.append(file)

# Check test count
print("\n\nChecking test count...")
try:
    spec = importlib.util.spec_from_file_location("test_module", "test_model.py")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Count test functions
    test_count = sum(1 for name in dir(test_module) if name.startswith('test_'))
    print(f"Found {test_count} tests in test_model.py")
    
    if test_count == 8:
        print("✅ All 8 tests present (including MAE and R² tests)")
    else:
        print(f"⚠️  Expected 8 tests but found {test_count}")
        print("Missing MAE and R² tests - update test_model.py!")
        
except Exception as e:
    print(f"❌ Could not check tests: {e}")

# Check for MAE and R2 functions in train.py
print("\n\nChecking train.py for metrics functions...")
try:
    with open("train.py", "r") as f:
        train_content = f.read()
        
    has_mae = "calculate_mae" in train_content
    has_r2 = "calculate_r2" in train_content
    
    if has_mae and has_r2:
        print("✅ MAE and R² calculation functions found")
    else:
        if not has_mae:
            print("❌ Missing calculate_mae function")
        if not has_r2:
            print("❌ Missing calculate_r2 function")
except:
    print("❌ Could not check train.py")

# Summary
print("\n" + "="*60)
if missing_files:
    print("❌ SETUP INCOMPLETE - Missing files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\nMake sure you have all the updated files from Claude!")
else:
    print("✅ All files present!")
    print("\nNext steps:")
    print("1. Run: python verify_tests.py")
    print("2. Run: python demo_all.py")
    print("3. Or follow the manual workflow in QUICK_REFERENCE.md")