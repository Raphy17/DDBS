from dbs import *

def create_db(path):
    db = Database(path)
    db.delete_table("table_un")
    db.create_table("table_un")
    db.fill_table("table_un", 1, 3000)
    return db


test = create_db("0dbs.db")
# create_db("1dbs.db")
# create_db("2dbs.db")
# create_db("3dbs.db")
print(test.get_table("table_un"))