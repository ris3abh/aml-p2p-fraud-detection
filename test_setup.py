#!/usr/bin/env python
"""Quick test to verify project setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.data_loader import load_paysim_data, get_basic_info

def main():
    print("Testing AML P2P Fraud Detection Setup...\n")
    
    # Test data loading
    try:
        df = load_paysim_data()
        print("\n✓ Data loading successful!")
        
        # Get basic info
        info = get_basic_info(df)
        print("\nDataset Summary:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        print("\n✓ All tests passed! You're ready to start the EDA phase.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Please check that the PaySim dataset is in data/raw/")

if __name__ == "__main__":
    main()
