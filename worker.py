from dbs import *

class Worker():
    counter_w = 0
    def __init__(self):
        self.db = self.create_db("dbs.db")
        self.counter_w += 1

    def create_db(self, path):
        db = Database(path)
        db.create_table("table_un")
        db.fill_table("table_un", 1, 1000)
        return db


worker_1 = Worker()
print(worker_1.db.get_k_random_sample("table_un", 100))
