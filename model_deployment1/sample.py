import os
import psycopg2
from sqlalchemy import create_engine
import mlflow
from mlflow import tracking
from utils.utils import get_best_run_id
from config.config import EXPERIMENT_NAME,MODEL_NAME
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

if __name__=="__main__":
    #get_best_run_id(EXPERIMENT_NAME)

    # Connect to the tracking server
    client = tracking.MlflowClient()

    try:
        # List experiments to confirm connection to the server
        experiments = client.search_experiments(f"name:{EXPERIMENT_NAME}")
        if experiments:
            print("Successfully connected to the MLflow tracking server!")
            print(f"Experiments: {[exp.name for exp in experiments]}")
        else:
            print("No experiments found.")
    except Exception as e:
        print(f"Failed to connect to the MLflow tracking server. Error: {e}")

    # Delete a registered model by name
    client.delete_registered_model(name=MODEL_NAME)

