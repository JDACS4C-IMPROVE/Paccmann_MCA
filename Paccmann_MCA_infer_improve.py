import pickle
import os
from pathlib import Path
from test_paccmann import main
# [Req] IMPROVE/CANDLE imports
# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

from model_params_def import infer_params # [Req]


filepath = Path(__file__).resolve().parent # [Req]

def run(params):
    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"], model_dir=params["input_model_dir"]) # [Req]

    params['train_data'] = Path(params["input_data_dir"]) / str('train_y_data'+'.csv')
    params['val_data'] = Path(params["input_data_dir"]) / str('val_y_data'+'.csv')
    params['gep_filepath'] = Path(params["input_data_dir"]) / params['gep_filepath']
    params['smi_filepath'] =Path(params["input_data_dir"]) / params['smi_filepath']
    params['gene_filepath'] = Path(params["input_supp_data_dir"]) / 'Data' / params['gene_filepath']
    params['smiles_language_filepath'] = Path(params["input_supp_data_dir"]) / 'Data' / params['smiles_language_filepath']
    params['modelpath'] = modelpath

    test_true, test_pred = main(params)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )
    return test_scores


def candle_main():
    additional_definitions = infer_params
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="Paccmann_MCA_default_model_csa.txt",
        additional_definitions=additional_definitions
    )

    with open(os.path.join(params['input_model_dir'], 'final_params.pickle'), 'rb') as handle:
        b = pickle.load(handle)
    params['smiles_vocabulary_size'] = int(b['smiles_vocabulary_size'])

    test_scores = run(params)
    print("\nFinished model inference.")
    
if __name__ == "__main__":
    candle_main()

