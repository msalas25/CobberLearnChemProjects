
import pandas as pd

csv_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_results.csv"

df = pd.read_csv(csv_file)

print("\n--- ERa DATA SUMMARY ---")
print("Total records:", len(df))
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values by column:")
print(df.isnull().sum())

print("\nUnique values by column:")
print(df.nunique())
print("\n--- HITC VALUE COUNTS ---")
print(df["hitc"].value_counts().head(10))

print("\n--- HITC SUMMARY ---")
print(df["hitc"].describe())
print(df["model"].value_counts())


import gzip
import re
import csv

sql_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\invitrodb_v4_3.sql.gz"
output_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERb_results.csv"

with gzip.open(sql_file, "rt", encoding="utf-8", errors="ignore") as f, \
     open(output_file, "w", newline="", encoding="utf-8") as out:

    writer = csv.writer(out)
    writer.writerow(["m5id", "m4id", "aeid", "model", "hitc", "fitc"])

    for line in f:
        if line.startswith("INSERT INTO `mc5`") and ",2115," in line:
            rows = re.findall(
                r"\((\d+),(\d+),2115,'([^']*)',([^,]*),([^,]*),",
                line
            )
            for row in rows:
                writer.writerow([row[0], row[1], 2115, row[2], row[3], row[4]])
            print("Records saved:", len(rows))
            break

print("Finished!")

print("\n--- ERb DATA SUMMARY ---")

er_b_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERb_results.csv"
er_b = pd.read_csv(er_b_file)

print("Total records:", len(er_b))
print("Duplicate rows:", er_b.duplicated().sum())

print("Missing values:")
print(er_b.isnull().sum())

print("Model counts:")
print(er_b["model"].value_counts())


import gzip
import re
import csv

sql_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\invitrodb_v4_3.sql.gz"
output_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERb_viability_results.csv"

with gzip.open(sql_file, "rt", encoding="utf-8", errors="ignore") as f, \
     open(output_file, "w", newline="", encoding="utf-8") as out:

    writer = csv.writer(out)
    writer.writerow(["m5id", "m4id", "aeid", "model", "hitc", "fitc"])

    for line in f:
        if line.startswith("INSERT INTO `mc5`") and ",2116," in line:
            rows = re.findall(
                r"\((\d+),(\d+),2116,'([^']*)',([^,]*),([^,]*),",
                line
            )
            for row in rows:
                writer.writerow([row[0], row[1], 2116, row[2], row[3], row[4]])
            print("Records saved:", len(rows))
            break

print("Finished!")


print("\n--- ERb VIABILITY DATA SUMMARY ---")

viability_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERb_viability_results.csv"
viability = pd.read_csv(viability_file)

print("Total records:", len(viability))
print("Duplicate rows:", viability.duplicated().sum())

print("Missing values:")
print(viability.isnull().sum())

print("Model counts:")
print(viability["model"].value_counts())



print("\n--- CHEMICAL TABLE COLUMNS ---")

with gzip.open(sql_file, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("CREATE TABLE `chemical`"):
            print(line)
            for next_line in f:
                print(next_line, end="")
                if next_line.strip().startswith(");"):
                    break
            break