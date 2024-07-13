# Create the boxplot for chronological age vs predicted age
# with gradient for the associated mean absolute error
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mtl
import matplotlib.pyplot as plt
# from matplotlib.patches import PathPatch
import matplotlib.lines as mlines

import psycopg2
import os
# from psycopg2 import Error
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_USER = 'DB_USER'
DB_PASSWORD = 'DB_PASSWORD'
DB_HOST = 'DB_HOST'
DB_PORT = 'DB_PORT'
DB_DATABASE = 'DB_DATABASE'

class PostgresManager:
    """ Manages the Postgres connection and methods to manipulate the database.
    """

    @staticmethod
    def get_database_uri(default_db=False):
        """ Build the database uri.
        """
        return 'postgresql://{}:{}@{}:{}/{}'.format(
            os.getenv(DB_USER),
            os.getenv(DB_PASSWORD),
            os.getenv(DB_HOST),
            os.getenv(DB_PORT),
            '' if default_db else os.getenv(DB_DATABASE)
        )

    def __init__(self, default_db=False, isolation_level=None):
        self.default_db = default_db
        self.isConnected = False
        self.isolation_level = isolation_level
    
    def __enter__(self):
        """ Sets up the connection to the postgres database.
        """
        self.connection = psycopg2.connect(
            self.get_database_uri(default_db=self.default_db))
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

    def run_sql(self, statement, parameters=None, fetch_one=False, fetch_all=False):
        self.cursor.execute(statement, parameters)
        self.connection.commit()

        if fetch_one:
            result = self.cursor.fetchone()
            return result[0] if result else None
        elif fetch_all:
            return self.cursor.fetchall()

model_id = ""
round = ""
output = {
    "age_gap": {},
    "brain_age": {},
}

with PostgresManager(default_db=True) as pg:
    output = pg.run_sql(f"SELECT predictions FROM runs WHERE model_id='{model_id}' AND round={round};")

age_gap = output["age_gap"]
brain_age = output["brain_age"]

by_age = []
age_ = []
pred_age_ = []
error_ = []

pred_age_by_age = {}
error_by_age = {}

min_age = 41
max_age = 92
sections = 5
for id in age_gap.keys():
    if id not in brain_age:
        print("Error")
        break
    age = brain_age[id] + age_gap[id]
    section = sections * (int(age)//sections)
    section = int(age)
    if section < min_age:
        section = min_age
    if section > max_age:
        section = max_age

    age_.append(section)
    pred_age_.append(brain_age[id])
    error_.append(-age_gap[id])
    
    if section not in pred_age_by_age:
        pred_age_by_age[section] = [] 
        error_by_age[section] = [] 
    pred_age_by_age[section].append(brain_age[id])
    error_by_age[section].append(abs(age_gap[id]))

for pag in pred_age_by_age:
    print(str(pag) + "      " + str(len(pred_age_by_age[pag])))

data = {'chronological age (years)': age_, 'predicted age (years)': pred_age_}
df = pd.DataFrame(data)
df.sort_values('chronological age (years)', ascending=True, inplace=True)

fig, ax = plt.subplots()


line = mlines.Line2D([0, 1], [0, 1], color='grey', linestyle='dashed',zorder=0)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

n_colors = 20
pal = sns.color_palette("coolwarm", n_colors)
pal_hexc = list(pal.as_hex())
print(pal_hexc)
palette = {}
for section in error_by_age:
    col_ch = int(np.average(error_by_age[section])/0.5)
    if col_ch > (n_colors - 1):
        col_ch = n_colors - 1
    palette[int(section)] = pal_hexc[col_ch]
sns.boxplot(x="chronological age (years)", y="predicted age (years)", palette=palette, data=df, showfliers = False, dodge=False)

ax.set_xlabel(ax.get_xlabel(), fontdict={'weight': 'bold'}, fontsize=14, labelpad=15)
ax.set_ylabel(ax.get_ylabel(), fontdict={'weight': 'bold'}, fontsize=14, labelpad=15)

xticks=ax.xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i%2==1:
        xticks[i].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=11)
plt.setp(ax.get_yticklabels(), fontsize=11)

plt.ylim(min_age, max_age)
fig.set_figwidth(10)
fig.set_figheight(10)

plt.show()
