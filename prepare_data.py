# prepare_data.py (Version 5 - Final Robust Version)
import pandas as pd
import pycountry

print("--- Phase 1 (Final): Data Preparation ---")
print("This script will robustly handle the mixed-type voting data.")

# --- Step 1: Load Raw Data (without specifying dtype) ---
print("\nStep 1: Loading raw UN voting data as is...")
try:
    # We remove the dtype argument to let pandas handle the mixed types initially
    df = pd.read_csv('UN_votes.csv')
    print(f" - Initial rows loaded: {len(df)}")
except FileNotFoundError:
    print("\n[ERROR] 'UN_votes.csv' not found. Please check file name and location.")
    exit()

# --- Step 2: Robustly Clean and Remap Votes ---
print("Step 2: Cleaning and remapping votes from the mixed-type column...")

# To handle all cases ('Y', 'y', 'Yes', 1, 1.0), we convert the column to strings and then lowercase.
# We fill any potential empty values with a placeholder that won't match anything.
df['vote_str'] = df['original_vote'].fillna('unknown').astype(str).str.lower()

# This is our master mapping dictionary. It handles all known variations.
vote_mapping = {
    # Yes Votes
    '1.0': 1.0, '1': 1.0, 'y': 1.0, 'yes': 1.0, 'for': 1.0,
    # No Votes
    '3.0': -1.0, '3': -1.0, 'n': -1.0, 'no': -1.0, 'against': -1.0,
    # Neutral/Abstain/Absent Votes
    '2.0': 0.0, '2': 0.0, 'a': 0.0, 'abstain': 0.0,
    '8.0': 0.0, '8': 0.0, 'b': 0.0, 'absent': 0.0
}
df['vote_numeric'] = df['vote_str'].map(vote_mapping)

# Remove any rows where the vote still couldn't be mapped
df.dropna(subset=['vote_numeric'], inplace=True)
print(f" - Rows remaining after cleaning votes: {len(df)}")

if len(df) == 0:
    print("\n[FATAL ERROR] All rows were removed after vote mapping. The vote values might be unexpected.")
    exit()

# --- Step 3: Extract Year and Add Standard ISO Country Codes ---
print("Step 3: Extracting year and adding standard ISO country codes...")
df['year'] = pd.to_datetime(df['meeting_date'], errors='coerce').dt.year
df.dropna(subset=['year'], inplace=True)
df['year'] = df['year'].astype(int)

def name_to_iso(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except (AttributeError, KeyError):
        if "Korea, Republic of" in country_name: return "KOR"
        if "United States" in country_name: return "USA"
        if "United Kingdom" in country_name: return "GBR"
        if "Russia" in country_name: return "RUS"
        if "Iran" in country_name: return "IRN"
        if "Vietnam" in country_name: return "VNM"
        if "Bolivia" in country_name: return "BOL"
        if "Venezuela" in country_name: return "VEN"
        if "Tanzania" in country_name: return "TZA"
        if "Syrian" in country_name: return "SYR"
        return None

df['iso_alpha'] = df['member_state'].apply(name_to_iso)
df.dropna(subset=['iso_alpha'], inplace=True)
print(f" - Rows remaining after mapping countries: {len(df)}")

if len(df) == 0:
    print("\n[FATAL ERROR] All rows were removed after the country mapping step. Check the name_to_iso function.")
    exit()

# --- Step 4: Pivot the Data ---
print("Step 4: Pivoting data frame...")
df_pivoted = df.pivot_table(
    index=['year', 'resolution_id'],
    columns='iso_alpha',
    values='vote_numeric'
)
df_pivoted.fillna(0, inplace=True)
print(" - Data successfully pivoted.")

# --- Step 5: Save the Processed Data ---
df_pivoted.to_csv('un_voting_pivoted.csv')
print("\n--- Phase 1 Complete! ---")
print(f"Successfully created 'un_voting_pivoted.csv' with {len(df_pivoted)} rows.")