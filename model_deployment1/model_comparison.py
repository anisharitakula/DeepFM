import os
import mlflow
import json
from config.config import EXPERIMENT_NAME,MODEL_NAME

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

def get_best_model_in_experiment(experiment_name):
    """
    Retrieve the best model from an MLflow experiment based on a performance metric
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Assuming we're using val_loss as the key metric - adjust as needed
    best_run = runs.sort_values('metrics.final_loss', ascending=False).iloc[0]
    
    return {
        'run_id': best_run['run_id'],
        'artifact_uri': best_run['artifact_uri'],
        'val_loss': best_run['metrics.final_loss']
    }

def get_production_model():
    """
    Retrieve the current production model information
    """
    try:
        client = mlflow.tracking.MlflowClient()
        production_models = client.search_model_versions("stage='Production'")
        
        if production_models:
            return production_models[0]
        return None
    except Exception as e:
        print(f"Error retrieving production model: {e}")
        return None

def compare_models(new_model, production_model):
    """
    Compare new model performance with production model
    """
    if not production_model:
        return True  # No existing production model
    
    # Compare accuracy or your preferred performance metric
    new_accuracy = new_model['val_loss']
    prod_accuracy = float(production_model.source.split('/')[-2])  # Adjust based on how you track metrics
    
    return new_accuracy > prod_accuracy

def main():
    experiment_name = EXPERIMENT_NAME
    
    # Get the best model from the latest experiment
    new_model = get_best_model_in_experiment(experiment_name)
    
    # Get current production model
    production_model = get_production_model()
    
    # Check if new model should be deployed
    is_deployable = compare_models(new_model, production_model)
    
    # Write deployment status to a file
    with open('deployment_status.txt', 'w') as f:
        f.write('true' if is_deployable else 'false')
    
    if is_deployable:
        # Write model URI for Docker build
        with open('model_uri.txt', 'w') as f:
            f.write(new_model['artifact_uri'])
        
        # Register the new model as production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_model['run_id'],
            stage='Production',
            archive_existing_versions=True
        )

if __name__ == '__main__':
    main()