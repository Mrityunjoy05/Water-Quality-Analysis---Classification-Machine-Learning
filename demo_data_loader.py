import yaml
import pandas as pd
import numpy as np
import os
from pathlib import Path
from core.data_loader import DataLoader

# --- GLOBAL CONSTANTS ---
# Using Uppercase for constants prevents naming conflicts inside functions
DEFAULT_CONFIG_FOLDER = "config"
DEFAULT_CONFIG_FILE = "config.yaml"

def load_config(config_path=None):
    """
    Safely load the YAML configuration file using Pathlib.
    """
    if config_path is None:
        # Joining path segments using the / operator
        config_path = Path(DEFAULT_CONFIG_FOLDER) / DEFAULT_CONFIG_FILE
    else:
        config_path = Path(config_path)

    print(f"🔍 Searching for config at: {config_path.absolute()}")

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")
        
    # We use 'f' here to avoid conflict with the global 'file' name or built-ins
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print(f"✅ Configuration loaded successfully!")
            return config
        except yaml.YAMLError as exc:
            raise ValueError(f"❌ Error parsing YAML file: {exc}")

def create_dummy_data_if_needed(config):
    """
    Ensures the raw data path exists. If not, generates a sample CSV.
    """
    raw_path = Path(config['data']['raw_data_path'])
    
    # Create the directory structure (data/raw/) if it doesn't exist
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print(f"⚠️ Raw data not found at {raw_path}. Generating dummy dataset...")
        
        # Sample data reflecting Water Quality parameters
        data = {
            'pH': [7.2, 8.1, np.nan, 6.9, 7.5, 8.0, 7.1],
            'DO': [6.0, 5.2, 4.8, np.nan, 7.0, 5.5, 6.2],
            'BOD': [1.5, 2.0, 4.5, 1.2, 2.1, 3.0, 1.8],
            'Nitrate': [0.4, 1.1, 2.5, 0.2, 0.9, 1.5, 0.6],
            'STN Code': [1, 2, 3, 4, 5, 6, 7],
            config['data']['target_column']: ['A', 'B', 'C', 'A', 'B', 'C', 'A'] 
        }
        df = pd.DataFrame(data)
        
        # Save with the encoding specified in config
        encoding = config['data'].get('encoding', 'cp1252')
        df.to_csv(raw_path, index=False, encoding=encoding)
        print(f"📝 Dummy data created at: {raw_path}")
        return True
    
    return False

def run_demo():
    print(f"\n{'='*60}")
    print("🚀 WATER QUALITY DATA LOADER - EXPERT DEMO")
    print(f"{'='*60}")

    try:
        # 1. Load configuration
        config = load_config()

        # 2. Setup mock environment
        created_temp = create_dummy_data_if_needed(config)

        # 3. Initialize the DataLoader class
        print("\n🔹 Step 1: Initializing DataLoader...")
        loader = DataLoader(config)

        # 4. Load the data
        print("\n🔹 Step 2: Loading Data...")
        df = loader.load_data()
        
        # 5. Show Summary Statistics
        print("\n🔹 Step 3: Generating Summary...")
        loader.print_data_summary()

        # 6. Identify Numeric vs Categorical
        print("\n🔹 Step 4: Analyzing Column Types...")
        loader.get_column_types()
        
        # 7. Check Class Balance
        print("\n🔹 Step 5: Checking Target Distribution...")
        loader.get_target_distribution()

        # 8. Test the Saving Functionality
        print("\n🔹 Step 6: Testing Processed Data Export...")
        test_file = "demo_processed_output.csv"
        loader.save_processed_data(df, test_file)
        
        # 9. Test Loading back the saved data
        print("\n🔹 Step 7: Testing Load of Processed Data...")
        df_loaded = loader.load_processed_data(test_file)
        print(f"   ✅ Successfully verified {len(df_loaded)} rows.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING DEMO: {str(e)}")
        # For professional debugging, uncomment the line below:
        # import traceback; traceback.print_exc()

    finally:
        print(f"\n{'='*60}")
        print("🏁 DEMO PROCESS COMPLETED")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    run_demo()