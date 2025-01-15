import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class DBEngine:
    def __init__(self):
        self.connection = self.connect()
        self.cursor = self.connection.cursor()

    @staticmethod
    def connect():
        try:
            # Assuming you have defined the database credentials in your .env file
            connection = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
            )
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def __del__(self):
        # Close cursor and connection when the object is deleted
        if self.connection:
            self.cursor.close()
            self.connection.close()


class IMDBDBTable:
    table_name = 'IMDBData'
    columns = ['rate', 'length']  # Example columns, add more as needed

    def __init__(self):
        self.db_connection = DBEngine()

    def create_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            rate VARCHAR(255),
            length INT
        )
        """
        self.db_connection.cursor.execute(query)
        self.db_connection.connection.commit()  # Fix commit typo
        print('Table IMDB created, or already exists.')

    def insert_data(self, df):
        self.create_table()
        for _, row in df.iterrows():
            columns = sql.SQL(', ').join(map(sql.Identifier, self.columns))
            values = sql.SQL(', ').join(map(sql.Placeholder, self.columns))
            query = sql.SQL(f'INSERT INTO {self.table_name} ({columns}) VALUES ({values})')
            self.db_connection.cursor.execute(query, row.to_dict())

        self.db_connection.connection.commit()  # Fix commit typo
        print('Data inserted.')


# Example usage (assuming df is a pandas DataFrame):
if __name__ == "__main__":
    # Example DataFrame (replace with actual data)
    data = {'rate': ['8.7', '9.2'], 'length': [120, 150]}
    df = pd.DataFrame(data)

    imdb_table = IMDBDBTable()
    imdb_table.insert_data(df)
