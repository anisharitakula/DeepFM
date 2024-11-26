import os
import psycopg2
from sqlalchemy import create_engine
import mlflow
from config.config import EXPERIMENT_NAME
import pandas as pd

def test_postgres_connection():
    try:
        # PostgreSQL Connection Parameters
        conn_params = {
            'dbname': 'mlflow_db',
            'user': 'inzaghianish',
            'password': 'Postgresql2289',
            'host': 'localhost',
            'port': '5432'
        }
        
        # Psycopg2 Direct Connection Test
        print("Testing direct psycopg2 connection...")
        conn = psycopg2.connect(**conn_params)
        print("Psycopg2 Direct Connection Successful!")
        conn.close()
        
        # SQLAlchemy Connection Test
        print("Testing SQLAlchemy connection...")
        sqlalchemy_uri = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        engine = create_engine(sqlalchemy_uri)
        
        with engine.connect() as connection:
            print("SQLAlchemy Connection Successful!")
        
        # MLflow Tracking URI Test
        print("Testing MLflow Tracking URI...")
        mlflow.set_tracking_uri(sqlalchemy_uri)
        mlflow_client = mlflow.tracking.MlflowClient()
        experiments = mlflow_client.list_experiments()
        print(f"Found {len(experiments)} existing experiments")
    
    except Exception as e:
        print(f"Connection Test Failed: {e}")
        import traceback
        traceback.print_exc()

