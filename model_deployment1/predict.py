import typer
from models.dfm import ModelDFM
import torch
import json
from preprocess.collate_fn import fm_collate_fn
from typing_extensions import Annotated
import ast

app=typer.Typer()

@app.command()
def predict(input: Annotated[str,typer.Option(help="tuple of user_id,movie_id and genre details") ]):
    parsed_input=ast.literal_eval(input)
    with open("config/config_data.json", "r") as file:
        config_data = json.load(file)
    
    with open("config/model_params.json", "r") as file:
        params_data = json.load(file)

    parsed_input[2]=parsed_input[2].split('|')
    parsed_input=tuple(parsed_input)
    model=ModelDFM(params_data['embed_dim'],config_data['unique_users']+config_data['unique_movies']+len(config_data['movie_genres_dict']))
    model.load_state_dict(torch.load("saved_models/deepfm_model.pth"))
    input_data=fm_collate_fn([parsed_input],**config_data,inference=True)
    model.eval()
    pred=model(input_data)
    print(float(pred))
    return float(pred)

if __name__=="__main__":
    app()
