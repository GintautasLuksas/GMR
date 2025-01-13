import psycopg2
from psycog
import os import sql
from dotenv import load_dotenv
class DBEngine:
    def __init__(self):
        self.connection = self.connect()
        self.cursor = self.connection.cursor()

        @statismethod
        def connect():
            try:
                connection = psycopg2.connect(


                )

                return connection

        def __dell__(self):
            if connection:
                self.cursor.close()
                self.connection.close()

class IMDBDBTable:
    table_name = 'IMDBData'
    column = {'rate', 'length'...}
    def __init__(self):
        self.db_connection = DBEngine()

    def create_table(self):
        query = f""""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
        id SERIAL PRIMARY KEY,
        rate VARCHAR(255),
        length INT)
        """
        self.db.connection.cursor.execute(query)
        self.db.connection.cursor.comit()
        print('Table IMDB created, or already exists.')

    def insert_data(self, df):
        self.create_table()
        for _, row in df.iterrows():
            columns = sql.SQL(', ').join(map(sql.Identifier, self.column))
            values = sql.SQL(', ').join(map(sql.Placeholder, df.keys()))
            query = sql.SQL(f'INSERT INTO {self.table_name} ({column}) VALUES ({values}')
            self.db_connection.cursor.execute(query, row.to_dict())

        self.db.conection.connection.comit()
        print('Data inserted.')