"""
Complete Pipeline Runner

Runs the entire prediction pipeline from data cleaning to final predictions.
"""
import sys
import subprocess

def run_step(script_name, description):
    """Run a pipeline step and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error in {script_name}")
        sys.exit(1)
    
    print(f"\n✓ {description} complete")
    return result.returncode

if __name__ == "__main__":
    print("="*80)
    print("FREIGHT RATE PREDICTION PIPELINE")
    print("="*80)
    print("\nThis will run the complete pipeline:")
    print("1. Data cleaning")
    print("2. Feature engineering") 
    print("3. Model training")
    print("4. Final predictions")
    print("\nEstimated time: 3-5 minutes")
    print("="*80)
    
    # Step 1: Data cleaning
    run_step('src/data_cleaning.py', 'Data Cleaning')
    
    # Step 2: Validation strategy
    run_step('src/validation_strategy.py', 'Validation Strategy')
    
    # Step 3: Feature engineering
    run_step('src/feature_engineering.py', 'Feature Engineering')
    
    # Step 4: Modeling
    run_step('src/modeling.py', 'Model Training')
    
    # Step 5: Final predictions
    run_step('src/predictions.py', 'Final Predictions')
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print("\nOutputs generated:")
    print("  ✓ validation_predictions.csv (12,000 predictions)")
    print("  ✓ data/december_chart_inputs.csv (31 December predictions)")
    print("  ✓ outputs/best_model.pkl (trained model)")
    print("  ✓ outputs/model_comparison.csv (model results)")
    print("\nReady for submission!")
