import sqlite3
import csv
import pandas as pd

# create a hypertension database by connecting python to SQLite 3
conn = sqlite3.connect(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\chronic_diseases.sqlite3")

# drop hypertension table if exist
try:
    conn.execute("DROP TABLE hypertension")
except sqlite3.OperationalError as e:
    print(e)
    
# create a sql table for hypertension
hypertension_sql = """CREATE TABLE hypertension(
                    id INTEGER NOT NULL,
                    age INTEGER NOT NULL,
                    obese TEXT NOT NULL,
                    bmi FLOAT NOT NULL,
                    wc FLOAT NOT NULL,
                    hc FLOAT NOT NULL,
                    whr FLOAT NOT NULL,
                    dbp INTEGER NOT NULL,
                    sbp INTEGER NOT NULL,
                    outcome TEXT NOT NULL
                    )"""
conn.execute(hypertension_sql)

# open hypertension csv file
fhand = open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\datasets\hypertension.csv")
hypertensionReader = csv.reader(fhand)
next(hypertensionReader)

# insert records into hypertension SQL table
for data in hypertensionReader:
    hypertensionSQL = """INSERT INTO hypertension(id, age, obese, bmi, wc, hc, whr, dbp, sbp, outcome)
                      VALUES(%d, %d, '%s', %f, %f, %f, %f, %d, %d, '%s')""" %(int(data[0]), int(data[1]), 
                      data[2], float(data[3]), float(data[4]), float(data[5]), float(data[6]),
                      int(data[7]), int(data[8]), data[9])
    
    cursor = conn.execute(hypertensionSQL)
conn.commit()

# retrieve to view the records from hypertension sql database
hyperTension = pd.read_sql("""SELECT * FROM hypertension""", conn)

conn.close()

