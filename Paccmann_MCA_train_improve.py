import pickle
import pickle
import os
from train_paccmann import main
import json
from pathlib import Path
import shutil

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

from model_params_def import train_params # [Req]

filepath = Path(__file__).resolve().parent # [Req]

def run(params):
    # [Req] Build model path
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"], model_dir=params["output_dir"])
    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]


    params['train_data'] = Path(params["input_dir"]) / str('train_y_data'+'.csv')
    params['val_data'] = Path(params["input_dir"]) / str('val_y_data'+'.csv')
    params['gep_filepath'] = Path(params["input_dir"]) / params['gep_filepath']
    params['smi_filepath'] =Path(params["input_dir"]) / params['smi_filepath']
    params['gene_filepath'] = Path(params["input_supp_data_dir"]) / 'Data' / params['gene_filepath']
    params['smiles_language_filepath'] = Path(params["input_supp_data_dir"]) / 'Data' / params['smiles_language_filepath']
    params['modelpath'] = modelpath


    val_true, val_pred, params_train = main(params)
    
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"]
    )
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    # Dump train_params into model outdir
    #Save as json
    with open(os.path.join(params['output_dir'], 'final_params.json') , "w") as file:
        json.dump(params_train, file, sort_keys=True, indent=4)

    return val_scores

def candle_main():
    additional_definitions = train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="Paccmann_MCA_default_model_csa.txt",
        additional_definitions=additional_definitions)
    
    val_scores = run(params)

if __name__ == "__main__":
    candle_main()

