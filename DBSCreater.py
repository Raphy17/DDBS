from dbs import *

def create_db(path):
    db = Database(path)
    db.create_table("table_uniform")
    db.create_table("table_pareto05")
    db.create_table("table_pareto15")
    db.fill_table_uniform("table_uniform", 1, 3000)
    db.fill_table_pareto("table_pareto05", 1, 3000, 0.5)
    db.fill_table_paret("table_pareto15", 1, 3000, 1.5)
    return db


create_db("0dbs.db")
create_db("1dbs.db")
create_db("2dbs.db")
create_db("3dbs.db")
create_db("4dbs.db")
create_db("5dbs.db")
create_db("6dbs.db")
create_db("7dbs.db")
create_db("8dbs.db")
create_db("9dbs.db")

