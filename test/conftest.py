import os
import sys

# Make the scripts in src/ importable the same way they import each other
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
