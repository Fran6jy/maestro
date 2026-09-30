"""
run_smoke_test.py
------------------
Place this file inside C:\\Users\\fran6\\Downloads\\maestro\\
Run it with:  python run_smoke_test.py

This script fixes its own Python path before importing anything,
so it works regardless of what directory you run it from.
"""
import sys
import os

# Fix the path: add the PARENT of this file's directory to sys.path
# This file lives in: Downloads\maestro\run_smoke_test.py
# So its parent is:   Downloads\
# That's what Python needs to find the 'maestro' package
THIS_FILE = os.path.abspath(__file__)
MAESTRO_DIR = os.path.dirname(THIS_FILE)       # Downloads\maestro
PARENT_DIR  = os.path.dirname(MAESTRO_DIR)     # Downloads

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

print(f"Script location:  {MAESTRO_DIR}")
print(f"Added to path:    {PARENT_DIR}")
print(f"Python:           {sys.version}")
print()

# Verify the package is now findable
try:
    import maestro
    print(f"✓ maestro package found at: {maestro.__file__}")
except ImportError as e:
    print(f"✗ Still can't find maestro: {e}")
    print()
    print("Your folder structure:")
    for item in os.listdir(MAESTRO_DIR):
        print(f"  {item}")
    sys.exit(1)

# Now run the smoke test
from maestro.tests.system_smoke import main
sys.exit(main())
