"""Quick fix script for Great Expectations issues"""
import os
import shutil

print("Fixing Great Expectations setup...")

# Remove existing GX directory if it exists
if os.path.exists("gx"):
    shutil.rmtree("gx")
    print("✅ Removed existing gx/ directory")

# Now run the validation setup
from validation import create_expectations, validate_data, check_model_performance
from data import create_sample_data

print("\nCreating fresh expectations...")
validator = create_expectations()

print("\nTesting validation with good data...")
test_df = create_sample_data(50)
results = validate_data(test_df)

print("\n✅ Validation setup complete!")
print("You can now run validation.py normally.")