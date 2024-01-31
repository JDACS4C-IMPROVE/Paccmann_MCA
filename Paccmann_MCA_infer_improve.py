import candle
import os
from pathlib import Path
from test_paccmann import main
# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics
from Paccmann_MCA_preprocess_improve import preprocess_params  # ap
from Paccmann_MCA_train_improve import metrics_list, model_train_params


file_path = os.path.dirname(os.path.realpath(__file__))

app_infer_params = []
model_infer_params = []
infer_params = app_infer_params + model_infer_params


def run(params):
    frm.create_outdir(outdir=params["infer_outdir"])
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    params['test_data'] = Path(params["ml_data_outdir"]) / str('test_'+params['y_data_suffix']+'.csv')
    params['gep_filepath'] = Path(params["ml_data_outdir"]) / params['gep_filepath']
    params['smi_filepath'] =Path(params["ml_data_outdir"]) / params['smi_filepath']
    params['gene_filepath'] = Path(params["ml_data_outdir"]) / params['gene_filepath']
    params['smiles_language_filepath'] = Path(params["ml_data_outdir"]) / params['smiles_language_filepath']

    test_true, test_pred = main(params)
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )
    return test_scores


def candle_main():
    additional_definitions = preprocess_params + model_train_params + infer_params
    params = frm.initialize_parameters(
        file_path,
        default_model="Paccmann_MCA_default_model_csa.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    with open("Paccmann_MCA_default_model_csa.txt", "r") as f:
        for line in f:
            if 'smiles_vocabulary_size' in line:
                vocab_size = line.split("=")[-1].strip("'\n ")
    params['smiles_vocabulary_size'] = int(vocab_size)

    test_scores = run(params)
    print("\nFinished model inference.")
    
if __name__ == "__main__":
    candle_main()

