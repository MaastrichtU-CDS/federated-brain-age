import os

import psycopg2

from federated_brain_age.constants import *

class PostgresManager:
    """ Manages the Postgres connection and methods to manipulate the database.
    """

    @staticmethod
    def get_database_uri(default_db=False, db_env_var=DB_DATABASE):
        """ Build the database uri.
        """
        return 'postgresql://{}:{}@{}:{}/{}'.format(
            os.getenv(DB_USER),
            os.getenv(DB_PASSWORD),
            os.getenv(DB_HOST),
            os.getenv(DB_PORT),
            '' if default_db else os.getenv(db_env_var)
        )

    def __init__(self, default_db=False, isolation_level=None, db_env_var=DB_DATABASE):
        self.default_db = default_db
        self.isConnected = False
        self.isolation_level = isolation_level
        self.db_env_var = db_env_var
    
    def __enter__(self):
        """ Sets up the connection to the postgres database.
        """
        self.connection = psycopg2.connect(
            self.get_database_uri(default_db=self.default_db, db_env_var=self.db_env_var))
        if self.connection:
            if self.isolation_level is not None:
                self.connection.set_isolation_level(self.isolation_level)
            self.cursor = self.connection.cursor()
            self.isConnected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Wraps up the connection and other settings when exiting.
        """
        if self.isConnected:
            self.connection.close()
            self.cursor.close()

    def create_database(self, database_name):
        """ Create a new database.
        """
        self.cursor.execute('CREATE DATABASE "{}";'.format(database_name))
        self.connection.commit()

    def create_sequence(self, name, start=1):
        """ Create a new sequence.
        """
        self.cursor.execute('DROP SEQUENCE IF EXISTS {};'.format(name))
        self.cursor.execute(
            'CREATE SEQUENCE IF NOT EXISTS {} AS BIGINT INCREMENT BY 1 START WITH {};'.format(
                name,
                start,
            )
        )
        self.connection.commit()

    def create_table(self, table_name, columns, ):
        """ Create a new table.
        """
        self.cursor.execute(f"""CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        );""")
        self.connection.commit()

    def drop_table(self, table):
        """ Drop a table from the database.
        """
        self.run_sql(f'DROP TABLE IF EXISTS {table};')

    def run_sql(self, statement, parameters=None, fetch_one=False, fetch_all=False):
        self.cursor.execute(statement, parameters)
        self.connection.commit()

        if fetch_one:
            result = self.cursor.fetchone()
            return result[0] if result else None
        elif fetch_all:
            return self.cursor.fetchall()

    def execute_file(self, path):
        """ Execute a file with a sql script.
        """
        self.cursor.execute(open(path, 'r').read())
        self.connection.commit()

    def copy_from_file(self, table, path):
        """ Insert data from a file.
        """
        with open(path, 'r') as data:
            self.cursor.copy_expert(f"COPY {table} FROM STDOUT WITH DELIMITER E'\t' NULL '' CSV HEADER QUOTE E'\b' ;", data)
            self.connection.commit()
