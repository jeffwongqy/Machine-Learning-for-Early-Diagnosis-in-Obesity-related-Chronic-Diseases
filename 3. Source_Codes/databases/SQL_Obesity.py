# import relevant libraries
import sqlite3
import csv
import pandas as pd

# create a obesity level database by connecting sqlite 3
conn = sqlite3.connect(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\chronic_diseases.sqlite3")

# drop obesity level table if do exist
try:
    conn.execute("DROP TABLE obesityLevel")
except sqlite3.OperationalError as e:
    print(e)

# create a sql table for obesity level
obesity_sql = """CREATE TABLE obesityLevel(
              gender TEXT NOT NULL,
              age INT NOT NULL,
              height FLOAT NOT NULL,
              weight FLOAT NOT NULL,
              family_history_with_overweight TEXT NOT NULL,
              caloric_food TEXT NOT NULL,
              vegetables INTEGER NOT NULL,
              number_meals INTEGER NOT NULL,
              food_between_meals INTEGER NOT NULL,
              smoke TEXT NOT NULL,
              water INTEGER NOT NULL,
              calories TEXT NOT NULL,
              activity INTEGER NOT NULL, 
              technology INTEGER NOT NULL,
              alcohol INTEGER NOT NULL,
              transportation TEXT NOT NULL,
              obesity_level INTEGER NOT NULL
              )"""
conn.execute(obesity_sql)

# open obesity csv file
fhand = open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\datasets\obesity.csv")
obesityReader = csv.reader(fhand)
next(obesityReader)

# insert records into obesity level SQL table
for data in obesityReader:
    obesitySQL = """INSERT INTO obesityLevel(gender, age, height, weight, family_history_with_overweight, caloric_food, vegetables, number_meals, food_between_meals, smoke, water, calories, activity, technology, alcohol, transportation, obesity_level)
                VALUES('%s', %d, %f, %f, '%s', '%s', %d, %d, %d, '%s', %d, '%s', %d, %d, %d, '%s', %d)"""%(data[0], int(data[1]), float(data[2]), 
                                                                                                           float(data[3]), data[4], data[5], 
                                                                                                           int(data[6]), int(data[7]), int(data[8]),
                                                                                                           data[9], int(data[10]), data[11], 
                                                                                                           int(data[12]), int(data[13]), int(data[14]), 
                                                                                                           data[15], int(data[16]))
    cursor = conn.execute(obesitySQL)
conn.commit()


# retrieve to view the records from obesity-level sql database
obesitylvl = pd.read_sql("""SELECT * FROM obesityLevel""", conn)
conn.close()