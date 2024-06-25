import candle
import pickle
import pickle
import os
from train_paccmann import main
import json
from pathlib import Path
import shutil
# IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics # TODO use comput_metrics in early stopping
from preprocess import model_preproc_params  # ap


# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = '/homes/brettin/Singularity/workspace/data_dir/'
file_path = os.path.dirname(os.path.realpath(__file__))

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = []

model_train_params = [
    {'name': 'gep_filepath',
     'type': str,
     'help': 'Path to the gene expression profile data.'
     },
    {'name': 'smi_filepath',
     'type': str,
     'help': 'Path to the SMILES data.'
     },
    {'name': 'gene_filepath',
     'type': str,
     'help': 'Path to a pickle object containing list of genes.'
     },
    {'name': 'smiles_language_filepath',
     'type': str,
     'help': 'Path to a pickle object a SMILES language object.'
     },

    {'name': 'drug_sensitivity_min_max',
     'type': bool,
     'help': '.....'
     },
    {'name': 'gene_expression_standardize',
     'type': bool,
     'help': 'Do you want to standardize gene expression data?'
     },
    {'name': 'augment_smiles',
     'type': bool,
     'help': 'Do you want to augment smiles data?'
     },
    {'name': 'smiles_start_stop_token',
     'type': bool,
     'help': '.....'
     },
    {'name': 'number_of_genes',
     'type': int,
     'help': 'Number of selected genes'
     },
    {'name': 'smiles_padding_length',
     'type': int,
     'help': 'Padding length for smiles strings'
     },
    {'name': 'filters',
     'type': list,
     'help': 'Size of filters'
     },
    {'name': 'multiheads',
     'type': list,
     'help': 'Size of multiheads for attention layer'
     },
    {'name': 'smiles_embedding_size',
     'type': int,
     'help': 'Size of smiles embedding'
     },
    {'name': 'kernel_sizes',
     'type': list,
     'help': 'Size of the kernels'
     },
    {'name': 'smiles_attention_size',
     'type': int,
     'help': 'Size of smiles attention'
     },
    {'name': 'embed_scale_grad',
     'type': bool,
     'help': '.....'
     },
    {'name': 'final_activation',
     'type': bool,
     'help': 'Is there a final activation?'
     },
    {'name': 'gene_to_dense',
     'type': bool,
     'help': '.....'
     },
    {'name': 'smiles_vocabulary_size',
     'type': int,
     'help': 'Size of smiles vocabulary'
     },
    {'name': 'number_of_parameters',
     'type': int,
     'help': 'Number of parameters'
     },
    {'name': 'drug_sensitivity_processing_parameters',
     'type': dict,
     'help': 'Parameters for drug sensitivity processing'
     },
    {'name': 'gene_expression_processing_parameters',
     'type': dict,
     'help': 'Parameters for processing gene expression data'
     }
]


def run(params):
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    params['train_data'] = Path(params["ml_data_outdir"]) / str('train_'+params['y_data_suffix']+'.csv')
    params['val_data'] = Path(params["ml_data_outdir"]) / str('val_'+params['y_data_suffix']+'.csv')
    params['gep_filepath'] = Path(params["ml_data_outdir"]) / params['gep_filepath']
    params['smi_filepath'] =Path(params["ml_data_outdir"]) / params['smi_filepath']
    params['gene_filepath'] = Path(params["ml_data_outdir"]) / params['gene_filepath']
    params['smiles_language_filepath'] = Path(params["ml_data_outdir"]) / params['smiles_language_filepath']
    params['modelpath'] = modelpath


    val_true, val_pred, params_train = main(params)
    
    # [Req] Save raw predictions in dataframe
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )
    # -----------------------------
    # [Req] Compute performance scores
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )
    # Dump train_params into model outdir
    with open(os.path.join(params['model_outdir'], 'final_params.pickle'), 'wb') as handle:
        pickle.dump(params_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return val_scores

def candle_main():
    additional_definitions = model_preproc_params + \
                             model_train_params + \
                             app_train_params
    params = frm.initialize_parameters(
        file_path,
        default_model="Paccmann_MCA_default_model.ini",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)

if __name__ == "__main__":
    candle_main()

