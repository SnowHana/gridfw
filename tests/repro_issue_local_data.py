import sys
import os

# Add src to python path if needed (might not be needed if run from root)
sys.path.append(os.path.join(os.getcwd(), "src"))

from grad_fw.data_loader import DatasetLoader

def verify_dataset(name):
    print(f"--- Testing {name} ---")
    try:
        loader = DatasetLoader()
        A_res, _ = loader.load(name)
        if A_res is not None:
            print(f"SUCCESS: Loaded {name} with shape {A_res.shape}")
        else:
            print(f"FAILURE: {name} loaded as None")
    except Exception as e:
        print(f"FAILURE: Exception loading {name}: {e}")

if __name__ == "__main__":
    verify_dataset("residential")
    verify_dataset("secom")
    verify_dataset("arrhythmia")
