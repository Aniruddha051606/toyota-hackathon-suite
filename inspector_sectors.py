import pandas as pd

try:
    df = pd.read_csv("ghost_lap.csv")

    if 'Laptrigger_lapdist_dls' in df.columns:
        print("\n--- Lap Distance Statistics ---")
        print("Use these values to define your track sectors.")

        # This will show us the min, max, 25%, 50%, and 75% marks
        print(df['Laptrigger_lapdist_dls'].describe())
    else:
        print("--- ERROR ---")
        print("Column 'Laptrigger_lapdist_dls' not found in ghost_lap.csv")

except FileNotFoundError:
    print("--- ERROR ---")
    print("File 'ghost_lap.csv' not found. Make sure it's in the same folder.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")