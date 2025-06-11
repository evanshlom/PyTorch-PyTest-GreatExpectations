"""Verify all tests are working correctly"""
import subprocess
import sys

print("Verifying PyTest setup...")
print("=" * 60)

# Run pytest with detailed output
result = subprocess.run(
    ["pytest", "test_model.py", "-v", "--tb=short"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Check results
if result.returncode == 0:
    print("\n✅ All tests passed!")
    
    # Count tests
    import re
    match = re.search(r'(\d+) passed', result.stdout)
    if match:
        num_tests = int(match.group(1))
        print(f"Total tests: {num_tests}")
        
        if num_tests == 8:
            print("✅ All 8 tests found (including MAE and R² tests)")
        elif num_tests == 6:
            print("⚠️  Only 6 tests found - missing MAE and R² tests")
            print("Make sure you have the updated test_model.py with all 8 tests")
else:
    print("\n❌ Some tests failed!")
    print("Return code:", result.returncode)

print("\nTest Summary:")
print("1. test_model_output_shape - Model outputs correct shape")
print("2. test_model_forward_pass - Forward pass works")
print("3. test_dataset_length - Dataset length is correct")
print("4. test_dataset_getitem - Dataset returns correct items")
print("5. test_create_sample_data - Sample data has correct structure")
print("6. test_normalize_features - Normalization works correctly")
print("7. test_calculate_mae - MAE calculation is accurate")
print("8. test_calculate_r2 - R² calculation is accurate")