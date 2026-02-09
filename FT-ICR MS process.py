import pandas as pd
import numpy as np

def filter_and_subtract_blank(bpm_file, blank_filtered_file, output_file):
    """
    Process BPM data:
    1. Apply the same filtering rules (N≤15, S≤1, H/C and O/C ratios)
    2. Remove duplicate m/z values
    3. Remove data with the same molecular formulas as in the Blank file
    4. Only retain molecular formulas unique to BPM
    """
    
    print("=" * 80)
    print("BPM Data Processing - Filter and Remove Molecular Formulas from Blank")
    print("=" * 80)
    
    # ========== Part 1: Read Data ==========
    print("\n[Step 1] Read data files")
    df_bpm = pd.read_csv(bpm_file)
    df_blank = pd.read_csv(blank_filtered_file)
    
    print(f"  BPM raw data: {len(df_bpm)} rows")
    print(f"  Blank filtered data: {len(df_blank)} rows")
    print(f"  Unique molecular formulas in Blank: {df_blank['sum formula'].nunique()}")
    
    # ========== Part 2: Filter BPM Data ==========
    print("\n[Step 2] Apply filtering rules to BPM")
    
    # Delete Unnamed columns
    df_BPM = df_BPM.loc[:, ~df_BPM.columns.str.contains('^Unnamed')]
    
    # Step 1: Filter N≤15, S≤1
    df_filtered = df_BPM[(df_BPM['N'] <= 15) & (df_BPM['S'] <= 1)].copy()
    removed_step1 = len(df_BPM) - len(df_filtered)
    print(f"  Element count filtering (N≤15, S≤1):")
    print(f"    Retained: {len(df_filtered)} rows, Removed: {removed_step1} rows")
    
    # Step 2: Calculate H/C and O/C ratios, and filter
    df_filtered = df_filtered[df_filtered['C'] > 0].copy()
    df_filtered['H_C_ratio'] = df_filtered['H'] / df_filtered['C']
    df_filtered['O_C_ratio'] = df_filtered['O'] / df_filtered['C']
    
    df_filtered = df_filtered[
        (df_filtered['H_C_ratio'] > 0.3) & 
        (df_filtered['H_C_ratio'] < 2.2) & 
        (df_filtered['O_C_ratio'] < 1.2)
    ].copy()
    removed_step2 = len(df_BPM) - removed_step1 - len(df_filtered)
    print(f"  Element ratio filtering (0.3 < H/C < 2.2, O/C < 1.2):")
    print(f"    Retained: {len(df_filtered)} rows, Removed: {removed_step2} rows")
    
    # Step 3: Remove duplicate m/z values
    df_filtered['heteroatom_count'] = df_filtered['N'] + df_filtered['O'] + df_filtered['S']
    df_filtered['abs_err_ppm'] = df_filtered['err ppm'].abs()
    
    duplicate_count_before = len(df_filtered) - df_filtered['Observed m/z'].nunique()
    
    df_filtered = df_filtered.sort_values(
        ['Observed m/z', 'heteroatom_count', 'abs_err_ppm']
    ).groupby('Observed m/z', as_index=False).first()
    
    print(f"  Remove duplicate m/z values:")
    print(f"    Before deduplication: {len(df_filtered) + duplicate_count_before} rows ({duplicate_count_before} duplicate rows)")
    print(f"    After deduplication: {len(df_filtered)} rows")
    
    # Delete auxiliary columns
    df_filtered = df_filtered.drop(columns=['H_C_ratio', 'O_C_ratio', 'heteroatom_count', 'abs_err_ppm'])
    
    # ========== Part 3: Remove Molecular Formulas Same as Blank ==========
    print("\n[Step 3] Remove molecular formulas identical to Blank")
    
    # Get the set of molecular formulas in Blank
    blank_formulas = set(df_blank['sum formula'].unique())
    print(f"  Number of molecular formulas in Blank: {len(blank_formulas)}")
    
    # BPM molecular formulas before filtering
    BPM_formulas_before = set(df_filtered['sum formula'].unique())
    print(f"  Number of molecular formulas in filtered BPM: {len(BPM_formulas_before)}")
    
    # Find common molecular formulas
    common_formulas = blank_formulas & BPM_formulas_before
    print(f"  Number of common molecular formulas: {len(common_formulas)}")
    
    # Remove common molecular formulas
    df_result = df_filtered[~df_filtered['sum formula'].isin(blank_formulas)].copy()
    
    # Find molecular formulas unique to BPM
    unique_to_BPM = BPM_formulas_before - blank_formulas
    print(f"  Molecular formulas unique to BPM: {len(unique_to_BPM)}")
    
    # Sort by Observed m/z
    df_result = df_result.sort_values('Observed m/z').reset_index(drop=True)
    
    # ========== Part 4: Save Results ==========
    print("\n[Step 4] Save results")
    df_result.to_csv(output_file, index=False)
    
    print(f"  Final retained rows: {len(df_result)}")
    print(f"  Results saved to: {output_file}")
    
    # ========== Part 5: Statistical Information ==========
    print("\n" + "=" * 80)
    print("Processing Completion Summary")
    print("=" * 80)
    print(f"\nCBPM Data Processing Flow:")
    print(f"  Raw data:           {len(df_BPM)} rows")
    print(f"  After element filtering:         {len(df_BPM) - removed_step1 - removed_step2} rows")
    print(f"  After deduplication:             {len(df_filtered) + len(df_result) - len(df_result)} rows")
    print(f"  After removing Blank duplicates:        {len(df_result)} rows")
    print(f"\nTotal removed data:          {len(df_BPM) - len(df_result)} rows ({(len(df_BPM) - len(df_result)) / len(df_BPM) * 100:.1f}%)")
    print(f"  Among which duplicates with Blank:    {len(df_filtered) - len(df_result)} rows")
    
    print(f"\nFinal Data Statistics:")
    print(f"  C range: {df_result['C'].min()} - {df_result['C'].max()}")
    print(f"  H range: {df_result['H'].min()} - {df_result['H'].max()}")
    print(f"  N range: {df_result['N'].min()} - {df_result['N'].max()}")
    print(f"  O range: {df_result['O'].min()} - {df_result['O'].max()}")
    print(f"  S range: {df_result['S'].min()} - {df_result['S'].max()}")
    print(f"  m/z range: {df_result['Observed m/z'].min():.4f} - {df_result['Observed m/z'].max():.4f}")
    
    # Element composition statistics
    print(f"\nElement Composition Statistics:")
    print(f"  Molecular formulas containing N: {(df_result['N'] > 0).sum()} ({(df_result['N'] > 0).sum() / len(df_result) * 100:.1f}%)")
    print(f"  Molecular formulas containing S: {(df_result['S'] > 0).sum()} ({(df_result['S'] > 0).sum() / len(df_result) * 100:.1f}%)")
    print(f"  Containing both N and S: {((df_result['N'] > 0) & (df_result['S'] > 0)).sum()}")
    
    # Error statistics
    print(f"\nError Statistics:")
    print(f"  err ppm average: {df_result['err ppm'].mean():.3f}")
    print(f"  err ppm standard deviation: {df_result['err ppm'].std():.3f}")
    print(f"  err ppm range: [{df_result['err ppm'].min():.3f}, {df_result['err ppm'].max():.3f}]")
    
    return df_result, common_formulas


if __name__ == "__main__":
    # File paths
    BPM_file = "D:/Bruker Data/FT-ICR MS data/12-25DATA/neg/1648A-NEG4.csv"
    blank_filtered_file = "D:/Bruker Data/FT-ICR MS data/12-25DATA/neg/Blank_filtered.csv"
    output_file = "D:/Bruker Data/FT-ICR MS data/12-25DATA/neg/1648A-NEG4-processed.csv"
    
    # Process data
    df_result, common_formulas = filter_and_subtract_blank(BPM_file, blank_filtered_file, output_file)
    
    # Display result preview
    print(f"\n" + "=" * 80)
    print("Result Preview (First 10 rows):")
    print("=" * 80)
    print(df_result.head(10).to_string())
    
    print(f"\nResult Preview (Last 10 rows):")
    print(df_result.tail(10).to_string())
    
    # Save common molecular formulas list
    common_formulas_file = "D:/Bruker Data/FT-ICR MS data/12-25DATA/pos/NBPM3-common_formulas.txt"
    with open(common_formulas_file, 'w') as f:
        f.write("List of common molecular formulas between Blank and BPM\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total {len(common_formulas)} common molecular formulas\n\n")
        for i, formula in enumerate(sorted(common_formulas), 1):
            f.write(f"{i}. {formula}\n")
    
    print(f"\nCommon molecular formulas list saved to: {common_formulas_file}")