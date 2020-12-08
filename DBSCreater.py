from DDBS.dbs import *

def create_db(path):
    db = Database(path)
    db.create_table("table_uniform")
    db.create_table("table_pareto05")
    db.create_table("table_pareto15")
    return db

def fill_table(db, table_name, size):
    db.fill_table_uniform("table_uniform", 1, size)
    db.fill_table_pareto("table_pareto05", 1, size, 0.5)
    db.fill_table_paret("table_pareto15", 1, size, 1.5)

