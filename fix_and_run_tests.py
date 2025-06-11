"""Quick fix for the test error and run all tests"""
import subprocess

print("🔧 Fixing test issues and running tests...")
print("=" * 60)

# First, let's make sure we have the right test
print("\nChecking test_model.py...")

# Read current test file
try:
    with open("test_model.py", "r") as f:
        content = f.read()
    
    # Check if we have the old version with integer tensors
    if 'torch.tensor([[1, 2, 3, 4, 5, 6, 7]])' in content:
        print("❌ Found old test version with integer tensors")
        print("✅ The fix has been provided - update test_model.py!")
        print("\nThe test_dataset_getitem function should use:")
        print('    features = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])')
        print('    targets = torch.tensor([100.0])')
        print("\nNot:")
        print('    features = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])  # integers!')
    
    # Count tests
    test_count = content.count('def test_')
    print(f"\nTotal tests found: {test_count}")
    
    if test_count < 8:
        print("⚠️  Missing MAE and R² tests - you have an old version!")
    
except Exception as e:
    print(f"Error reading test file: {e}")

# Now run the tests
print("\n" + "="*60)
print("Running pytest...")
print("="*60)

result = subprocess.run(
    ["pytest", "test_model.py", "-v"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Summary
if result.returncode == 0:
    print("\n✅ All tests passed!")
else:
    print("\n❌ Tests failed - make sure you have the updated files!")
    print("\nQuick fix:")
    print("1. Update test_model.py with float tensors (1.0 instead of 1)")
    print("2. Make sure data.py handles type conversion properly")
    print("3. Run: pytest test_model.py -v")