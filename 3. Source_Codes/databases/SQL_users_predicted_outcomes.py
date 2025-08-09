import sqlite3
from sqlite3 import Error


# define the name of the database
DBFILENAME = "usersPredictedOutcomes.db"

def create_connection(db_filename):
  try:
    conn = sqlite3.connect(db_filename)
    return conn
  except Error as e:
    print(e)
  return None


# creates the tables for the database
def create_tables():
  # create SQL table for users with predicted diabetes outcomes
  diab_pred_outcomes_table = """CREATE TABLE IF NOT EXISTS diab_pred_outcome (
                                        id INTEGER PRIMARY KEY,
                                        name TEXT NOT NULL,
                                        identification_no TEXT NOT NULL,
                                        email_address TEXT NOT NULL, 
                                        gender TEXT NOT NULL,
                                        age INTEGER NOT NULL,
                                        bmi FLOAT NOT NULL,
                                        glucose FLOAT NOT NULL,
                                        predicted_class INTEGER NOT NULL,
                                        predicted_proba FLOAT NOT NULL,
                                        predicted_outcome TEXT NOT NULL
                                        );"""
  
  bp_pred_outcomes_table = """CREATE TABLE IF NOT EXISTS bp_pred_outcome(
                                          id INTEGER PRIMARY KEY,
                                          name TEXT NOT NULL,
                                          identification_no TEXT NOT NULL,
                                          email_address TEXT NOT NULL,
                                          gender TEXT NOT NULL,
                                          age INTEGER NOT NULL,
                                          bmi FLOAT NOT NULL,
                                          sbp INTEGER NOT NULL,
                                          dbp INTEGER NOT NULL,
                                          predicted_class INTEGER NOT NULL,
                                          predicted_proba FLOAT NOT NULL,
                                          predicted_outcome TEXT NOT NULL);
                                          """
  
  obesity_pred_outcomes_table = """CREATE TABLE IF NOT EXISTS obesity_pred_outcome(
                                                id INTEGER PRIMARY KEY,
                                                name TEXT NOT NULL,
                                                identification_no TEXT NOT NULL,
                                                email_address TEXT NOT NULL,
                                                gender TEXT NOT NULL,
                                                age INTEGER NOT NULL,
                                                height FLOAT NOT NULL,
                                                weight FLOAT NOT NULL,
                                                family_history_obesity TEXT NOT NULL,
                                                having_caloric_food TEXT NOT NULL,
                                                having_fruit_vegetables TEXT NOT NULL,
                                                num_of_meals INTEGER NOT NULL,
                                                between_meals TEXT NOT NULL,
                                                smoking TEXT NOT NULL,
                                                drinking_water TEXT NOT NULL,
                                                monitor_calories TEXT NOT NULL,
                                                activity TEXT NOT NULL,
                                                usage_of_technology TEXT NOT NULL,
                                                drinking_alcohol TEXT NOT NULL,
                                                predicted_class INTEGER NOT NULL,
                                                predicted_proba FLOAT NOT NULL,
                                                predicted_outcome TEXT NOT NULL
                                                );"""

  
  try:
    conn = create_connection(DBFILENAME)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS diab_pred_outcome;")
    cur.execute("DROP TABLE IF EXISTS bp_pred_outcome;")
    cur.execute("DROP TABLE IF EXISTS obesity_pred_outcome;")
    cur.execute(diab_pred_outcomes_table)
    cur.execute(bp_pred_outcomes_table)
    cur.execute(obesity_pred_outcomes_table)
    conn.commit()
    conn.close()
  except Error as e:
    print(e)
    
if __name__ =='__main__':
    create_tables()
    