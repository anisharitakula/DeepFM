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

        production_models = client.search_model_versions(f"name='{MODEL_NAME}'")
        
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
    new_loss = new_model['val_loss']
    
    # Extract production model loss from tags or metrics
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(production_model.run_id)
    prod_loss = run.data.metrics.get('final_loss', float('inf'))
    
    return new_loss < prod_loss

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

        try:
            client.create_registered_model(name=MODEL_NAME)
        except mlflow.exceptions.RestException:
            print(f"Model {MODEL_NAME} already exists.")

        # Register the model
        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=new_model['artifact_uri'],
            run_id=new_model['run_id']
        )

        # Transition to Production stage
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage='Production',
            archive_existing_versions=True
        )

if __name__ == '__main__':
    main()