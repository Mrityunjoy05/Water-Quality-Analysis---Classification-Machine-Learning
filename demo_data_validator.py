import yaml
from pathlib import Path
from core.data_loader import DataLoader
from core.data_validation import DataValidator  

# Constants
CONFIG_PATH = Path("config") / "config.yaml"

def load_config_dict():
    """Helper to load config for the demo."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"❌ Could not find config at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def run_validation_demo():
    print(f"\n{'='*60}")
    print("🚀 WATER QUALITY DATA VALIDATION - DEMO")
    print(f"{'='*60}")

    try:
        # 1. Initialize Components
        config = load_config_dict()
        loader = DataLoader(config)
        validator = DataValidator()

        # 2. Load Data
        print("\n🔹 Step 1: Loading data for validation...")
        df = loader.load_data()

        if df is not None:
            # 3. Run General Validation
            # This triggers: empty check, duplicates, missing values, dtypes, and outliers
            print("\n🔹 Step 2: Running Comprehensive Validation...")
            is_valid = validator.validate_dataframe(df)

            # 4. Check Target Variable specifically
            # We get the target name from the config we loaded
            target_col = config['data']['target_column']
            print(f"\n🔹 Step 3: Checking Target Variable ('{target_col}')...")
            target_report = validator.check_target_distribution(df, target_col)

            # 5. Review the Final JSON Report
            print("\n🔹 Step 4: Final Validation Report Summary...")
            full_report = validator.get_validation_report()
            
            # Print a quick summary of the report dictionary
            print(f"   • Duplicate Rows Found: {full_report.get('n_duplicates', 0)}")
            print(f"   • Columns with Outliers: {len(full_report.get('outliers', {}))}")
            print(f"   • Target Classes Found: {target_report.get('n_classes')}")

    except Exception as e:
        print(f"\n❌ VALIDATION DEMO FAILED: {str(e)}")
        # import traceback; traceback.print_exc()

    finally:
        print(f"\n{'='*60}")
        print("🏁 VALIDATION DEMO COMPLETED")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    run_validation_demo()