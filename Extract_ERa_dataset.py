
import pandas as pd

file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_BLA_Agonist_ratio_extracted.csv"

df = pd.read_csv(file)

print("--- EXTRACTION CHECK ---")
print("Total records:", len(df))
print("Duplicate rows:", df.duplicated().sum())

print("\nAEID counts:")
print(df["aeid"].value_counts(dropna=False))

print("\nModel counts:")
print(df["model"].value_counts(dropna=False).head(20))

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))