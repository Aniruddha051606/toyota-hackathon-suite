import os
import pandas as pd

print("🔍 Searching for ANY telemetry files in subfolders...")
found_any = False

# Walk through all folders to find any CSV with "telemetry" in the name
for root, dirs, files in os.walk("."):
    for file in files:
        # We look for ANY csv that has 'telemetry' in the name
        if file.endswith(".csv") and "telemetry" in file.lower():
            found_any = True
            full_path = os.path.join(root, file)
            print(f"\n✅ FOUND FILE: {file}")
            print(f"   Full Path: {full_path}")
            
            try:
                print("   Loading data to calculate sectors...")
                # Load only the distance column
                df = pd.read_csv(full_path, usecols=['telemetry_name', 'telemetry_value'])
                
                # Filter for distance
                dist_df = df[df['telemetry_name'] == 'Laptrigger_lapdist_dls']
                dist_df['value'] = pd.to_numeric(dist_df['telemetry_value'], errors='coerce')
                
                print("\n--- Indianapolis Track Statistics ---")
                print(dist_df['value'].describe())
                
                # Get the clean relative path for your config
                rel_path = os.path.relpath(full_path).replace("\\", "/")
                print(f"\n👇 COPY THIS PATH FOR YOUR SCRIPTS 👇")
                print(f'csv_path = "{rel_path}"')
                
                # Stop after finding the first valid file to avoid confusion
                break 
            except Exception as e:
                print(f"   ⚠️ Found file but couldn't read it: {e}")

    if found_any:
        break

if not found_any:
    print("\n❌ Still no telemetry files found.")
    print("Here are the files in your current folder:")
    print(os.listdir("."))