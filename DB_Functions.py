import numpy as np
from typing import List, Dict, Any
import psycopg2
from scipy.stats import entropy
import pandas as pd

class DB_Functions:
    def connect_database( self , db_name , user ):
        '''
        Connects to the given database
        Returns a dict:
        conn       : db connection object
        count      : (int)
        '''
        conn = psycopg2.connect( database = db_name , user = user , password = "swaroopsk123z" ,
        host = "localhost" ,  port = "5432" )
        return( conn )

    def select_query( self , conn , query ):
        cur = conn.cursor()
        cur.execute( query )
        rows = cur.fetchall()
        return rows

    def fetch_data( self , conn , query , query_params = None ):
        return pd.read_sql_query( query , conn , params = query_params )

