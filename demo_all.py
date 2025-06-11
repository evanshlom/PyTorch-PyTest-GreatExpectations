"""Run complete demo in one script"""
import subprocess
import time
import os

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    time.sleep(1)  # Brief pause for readability
    return result.returncode == 0

# Main demo flow
print("🎬 PYTORCH STOCK PREDICTION DEMO")
print("=" * 60)

# 1. Train the model
if not os.path.exists("model.pth"):
    run_command("python train.py", "Training the Model")
else:
    print("\n✅ Model already trained (model.pth exists)")

# 2. Run unit tests
run_command("pytest test_model.py -v", "Running Unit Tests")

# 3. Fix and run validation
if not os.path.exists("gx"):
    run_command("python fix_validation.py", "Setting up Data Validation")
else:
    run_command("python validation.py", "Running Data Validation")

# 4. Show bad data
run_command("python data_bad.py", "Showing Bad Data Examples")

# 5. Test with bad data
run_command("python test_bad_data.py", "Testing Validation with Bad Data")

print("\n" + "="*60)
print("✅ DEMO COMPLETE!")
print("="*60)
print("\nKey Takeaways:")
print("1. PyTest ensures code correctness")
print("2. Great Expectations catches data quality issues")
print("3. Model metrics track performance (Loss, MAE, R²)")
print("4. Bad data is common - validation saves debugging time!")