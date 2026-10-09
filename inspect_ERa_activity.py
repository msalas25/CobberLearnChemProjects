
import pandas as pd

file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_BLA_Agonist_ratio_extracted.csv"

df = pd.read_csv(file)

print("--- DATASET OVERVIEW ---")
print("Total assay records:", len(df))
print("Unique m4id values:", df["m4id"].nunique())
print("Unique m5id values:", df["m5id"].nunique())

print("\n--- ACTIVITY CALLS (hitc) ---")
print(df["hitc"].describe())
print("\nMost common hitc values:")
print(df["hitc"].value_counts().head(20))

print("\n--- FIT STATUS (fitc) ---")
print(df["fitc"].value_counts(dropna=False).sort_index())

print("\n--- MODEL TYPES ---")
print(df["model"].value_counts(dropna=False))

print("\n--- ACTIVITY CALLS BY FIT STATUS ---")
print(pd.crosstab(df["fitc"], df["hitc"] == 1.0))
