import sqlite3
from sqlite3 import Error
import random



class Database:
    counter_db = 0
    counter_t = 0
    def __init__(self, path):
        self.counter_db += 1
        self.path = path
        self.conn = self.create_connection()


    def __str__(self):
        return "Database {}".format(self.counter_db)

    def create_connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path)
            print(connection)
            print("Connection to SQLite DB successful")
        except Error as e:
            print("Error: '{}' occurred".format(e))

        return connection

    def create_table(self, table_name):
        self.counter_t += 1
        sample_table = "CREATE TABLE IF NOT EXISTS {}(age INTEGER, loc_x INTEGER, loc_y INTEGER, dim_1 INTEGER," \
                       "dim_2 INTEGER, id INTEGER PRIMARY KEY, gender INTEGER);".format(table_name)
        self.execute_query(sample_table)
        return sample_table

    def execute_query(self, query):
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            self.conn.commit()
        except Error as e:
            print(e)

    def fill_table(self, table_name, gender, n):
        for i in range(n+1):
            age = random.randint(0, 100)
            x_loc = random.randint(0, 1000)
            y_loc = random.randint(0, 1000)
            dim_1 = random.randint(0, 1000)
            dim_2 = random.randint(0, 1000)
            insert_vals = "INSERT INTO {} VALUES({},{},{},{},{},{},{})".format(table_name, age, x_loc, y_loc, dim_1, dim_2, i, gender)
            self.execute_query(insert_vals)


    def execute_read_query(self, query):
        cursor = self.conn.cursor()
        result = []
        try:
            if type(query) == str:
                cursor.execute(str(query))
                result = cursor.fetchall()
            else:
                for el in query:
                    cursor.execute(el)
                    result.append(cursor.fetchall()[0])
            return tuple(result)
        except Error as e:
            print("The error '{}' occurred".format(e))
            return None


    def get_table(self, table_name):
        select_table = "SELECT * from {}".format(table_name)
        values = self.execute_read_query(select_table)
        return values

    def delete_table(self, table_name):
        query = "DROP TABLE {}".format(table_name)
        self.execute_query(query)

    def get_k_random_sample(self, table_name, k):
        queries = []
        for i in range(k):
            idx = random.randint(0,1000)
            queries.append("SELECT * from {} WHERE id = {}".format(table_name, idx))
        return self.execute_read_query(queries)


#
# db = Database("dbs.db")
# db.delete_table("type")
# table_type = db.create_table("type")
# db.fill_table("type", 1, 1000)
# print(db.get_table("type"))
# print(db.get_k_random_sample("type", 100))