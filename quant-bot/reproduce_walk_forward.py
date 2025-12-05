import sys
import os
sys.path.append(os.getcwd())

print("Attempting to import HistoricalDataLoader...")
try:
    from src.validation.walk_forward import HistoricalDataLoader
    print("HistoricalDataLoader imported successfully.")
except Exception as e:
    print(f"Failed to import HistoricalDataLoader: {e}")
    import traceback
    traceback.print_exc()
