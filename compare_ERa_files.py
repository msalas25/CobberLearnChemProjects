
import pandas as pd

old_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_results.csv"

new_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_BLA_Agonist_ratio_extracted.csv"

old = pd.read_csv(old_file)
new = pd.read_csv(new_file)

print("--- OLD ERa FILE ---")
print("Rows:", len(old))
print("Columns:", old.columns.tolist())

print("\n--- NEW ERa FILE ---")
print("Rows:", len(new))
print("Columns:", new.columns.tolist())

# Compare the IDs present in both files
if "m5id" in old.columns and "m5id" in new.columns:
    old_ids = set(old["m5id"].dropna())
    new_ids = set(new["m5id"].dropna())

    print("\n--- ID COMPARISON ---")
    print("Unique old IDs:", len(old_ids))
    print("Unique new IDs:", len(new_ids))
    print("IDs in both:", len(old_ids & new_ids))
    print("Only in old:", len(old_ids - new_ids))
    print("Only in new:", len(new_ids - old_ids))

if "hitc" in old.columns:
    print("\nOld hitc counts:")
    print(old["hitc"].value_counts(dropna=False).head(10))

if "hitc" in new.columns:
    print("\nNew hitc counts:")
    print(new["hitc"].value_counts(dropna=False).head(10))