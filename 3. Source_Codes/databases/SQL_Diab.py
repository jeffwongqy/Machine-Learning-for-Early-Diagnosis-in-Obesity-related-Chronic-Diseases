import sqlite3
import csv
import pandas as pd

# create a diabetes database by connecting python to SQLite 3
conn = sqlite3.connect(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\chronic_diseases.sqlite3")

# drop SQL table if exist
try: 
    conn.execute("DROP TABLE diabetes")
except sqlite3.OperationalError as e:
    print(e)
    

# create a sql table for diabetes 
diabetes_sql = """CREATE TABLE diabetes(
               ID INTEGER NOT NULL,
               No_Pation INTEGER NOT NULL,
               Gender TEXT NOT NULL,
               AGE INTEGER NOT NULL,
               Urea FLOAT NOT NULL,
               Cr FLOAT NOT NULL,
               HbA1c FLOAT NOT NULL,
               Chol FLOAT NOT NULL,
               TG FLOAT NOT NULL,
               HDL FLOAT NOT NULL,
               LDL FLOAT NOT NULL,
               VLDL FLOAT NOT NULL,
               BMI FLOAT NOT NULL,
               CLASS TEXT NOT NULL
                )"""
conn.execute(diabetes_sql)


# open diabetes csv file
fhand = open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\datasets\diab.csv")
diabetesReader = csv.reader(fhand)

# remove the header column
next(diabetesReader)

# insert records into SQL diabetes table 
for data in diabetesReader:
    diabetesSQL = """INSERT INTO diabetes (ID, No_Pation, Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI, CLASS)
                    VALUES(%d, %d, '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, '%s')""" %(int(data[0]), int(data[1]), (data[2]), int(data[3]),
                                                                                                float(data[4]), float(data[5]), float(data[6]),
                                                                                                float(data[7]), float(data[8]), float(data[9]),
                                                                                                float(data[10]), float(data[11]), float(data[12]),
                                                                                                data[13])
    
    cursor = conn.execute(diabetesSQL)
conn.commit()

# retrieve to view records from the diabetes SQL table
diabetes = pd.read_sql("""SELECT * FROM diabetes""", conn)
conn.close()