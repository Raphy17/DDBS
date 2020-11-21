import sqlite3
from sqlite3 import Error
import random
import numpy as np


class Database:
    counter_db = 0
    counter_t = 0
    def __init__(self, path):
        self.counter_db += 1
        self.path = path
        self.conn = self.create_connection()
        self.size = 3000


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
        except Exception as e:
            e



    def fill_table_uniform(self, table_name, gender, n):
        self.size = n
        for i in range(n):
            age = random.randint(0, 1000)
            x_loc = random.randint(0, 1000)
            y_loc = random.randint(0, 1000)
            dim_1 = random.randint(0, 1000)
            dim_2 = random.randint(0, 1000)
            insert_vals = "INSERT INTO {} VALUES({},{},{},{},{},{},{})".format(table_name, age, x_loc, y_loc, dim_1, dim_2, i, gender*(i % 2))
            self.execute_query(insert_vals)

    def fill_table_pareto(self, table_name, gender, n, z):
        values = []
        dim = 5
        for i in range(dim):
            x = (np.random.pareto(z, n)+1)
            values.append(x)

        for i in range(len(x)):
            t = []
            for d in range(dim):
                t.append(values[d][i])

            insert_vals = "INSERT INTO {} VALUES({},{},{},{},{},{},{})".format(table_name, t[0], t[1], t[2],t[3],t[4], i, gender*(i % 2))
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
            return list(result)
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

    def get_k_random_samples(self, table_name, k):
        queries = []
        rand_ints = []
        for i in range(k):
            idx = random.randint(0, self.size)
            while idx in rand_ints:
                idx = random.randint(0, self.size)
            rand_ints.append(idx)

            queries.append("SELECT * from {} WHERE id = {}".format(table_name, idx))
        return self.execute_read_query(queries)


#
# db = Database("dbs.db")
# #db.delete_table("type")
# table_type = db.create_table("type")
# db.fill_table("type", 1, 1000)
# print(db.get_table("type"))

