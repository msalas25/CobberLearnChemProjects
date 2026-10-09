
import gzip
import re
import csv

sql_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\invitrodb_v4_3.sql.gz"

# Use a NEW filename so your existing ERa_results.csv stays safe
output_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_BLA_Agonist_ratio_extracted.csv"

target_aeid = 785
total_records = 0

with gzip.open(sql_file, "rt", encoding="utf-8", errors="ignore") as f, \
     open(output_file, "w", newline="", encoding="utf-8") as out:

    writer = csv.writer(out)
    writer.writerow(["m5id", "m4id", "aeid", "model", "hitc", "fitc"])

    for line in f:
        if line.startswith("INSERT INTO `mc5`"):

            # Match records whose third field (aeid) is 785
            rows = re.findall(
                r"\((\d+),(\d+),785,'([^']*)',([^,]*),([^,]*),",
                line
            )

            for row in rows:
                writer.writerow([
                    row[0], row[1], target_aeid,
                    row[2], row[3], row[4]
                ])

            total_records += len(rows)

print("Finished extracting ER-alpha BLA data!")
print("AEID:", target_aeid)
print("Total records extracted:", total_records)
print("Saved to:", output_file)

if total_records == 0:
    print("WARNING: No records found. Check the SQL format and AEID.")