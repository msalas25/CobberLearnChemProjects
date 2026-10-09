
import gzip

sql_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\invitrodb_v4_3.sql.gz"

with gzip.open(sql_file, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("CREATE TABLE `chemical`"):
            print("Found the chemical table!")

            for next_line in f:
                print(next_line, end="")

                if next_line.lstrip().startswith(")"):                    break

            break