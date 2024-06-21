import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional
import shutil
from IMPROVE import framework as frm
from IMPROVE import drug_resp_pred as drp
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path
import candle


file_path = os.path.dirname(os.path.realpath(__file__))

# Model-specific params (Model: Paccmann_MCA)
model_preproc_params = [
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

# App-specific params (App: drug response prediction)
drp_preproc_params = [
    {"name": "x_data_canc_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "x_data_path",  # app;
     # "nargs": "+",
     "default":"x_data",
     "type": str,
     "help": "Path to x data",
    },
    {"name": "y_data_path",  # app;
     # "nargs": "+",
     "default":"y_data",
     "type": str,
     "help": "Path to y data",
    },
    {"name": "splits_path",  # app;
     # "nargs": "+",
     "default":"splits",
     "type": str,
     "help": "Path to splits",
    },
    {"name": "x_data_drug_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data_files",  # imp;
     # "nargs": "+",
     "type": str,
     "help": "List of output files.",
    },
    {"name": "canc_col_name",  # app;
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",  # app;
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]
preprocess_params = model_preproc_params + drp_preproc_params
req_preprocess_args = [ll["name"] for ll in preprocess_params]
req_preprocess_args.extend(["y_col_name", "model_outdir"])


def run(params):
    #params = frm.build_paths(params)  # paths to raw data
    #frm.create_outdir(outdir=params["ml_data_outdir"])
    params['x_data_path'] = Path(params['x_data_path'])
    params['y_data_path'] = Path(params['y_data_path'])
    params['splits_path'] = Path(params['splits_path'])

    print("\nLoading omics data...")
    oo = drp.OmicsLoader(params)
    print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get the needed canc x data
    
    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    print(dd)
    sm = dd.dfs['drug_SMILES.tsv']  # get the needed drug x data
    # Modify files to be compatible with Paccmann_MCA (Model specific modification)
    sm_new = pd.DataFrame(columns = ['SMILES', 'DrugID'])
    sm_new['SMILES'] = sm['canSMILES'].values
    sm_new['DrugID'] = sm.index.values
    
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        if split_file=='':
            continue
        # ---------------------------------
        # [Req] Load response data
        # ------------------------
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        df_response = rr.dfs["response.tsv"]
        # ------------------------
        # Retain (canc, drug) response samples for which omic data is available
        df_y, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                               ref_col=params["canc_col_name"])
        print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        df_y = df_y.rename(columns = {'improve_chem_id':'drug', 'improve_sample_id':'cell_line'}) # Model specfic change in column names
        df_y['IC50'] = df_y['auc']
        df_y.reset_index(inplace=True)
        frm.save_stage_ydf(df_y, params, stage)
        # [Req] Create data name
        #data_fname = frm.build_ml_data_name(params, stage)

    # Save SMILES as .smi format as required by the model (Model specific)
    if os.path.exists(os.path.join(Path(params["ml_data_outdir"]),'smiles.smi')): # Remove smiles.smi to prevent duplicates
        os.remove(os.path.join(Path(params["ml_data_outdir"]),'smiles.smi'))
    sm_new.to_csv(Path(params["ml_data_outdir"]) / "smiles.csv", index=False)
    newfile = os.path.join(Path(params["ml_data_outdir"]),'smiles.smi')
    file = os.path.join(Path(params["ml_data_outdir"]),'smiles.csv')
    with open(file,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  ## skip one line (the first one)
        for line in csv_reader:
            with open(newfile, 'a') as new_txt:    #new file has .txt extn
                txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
                txt_writer.writerow(line)   #write the lines to file`
    # Save Gene expression
    ge=ge.rename(columns={'improve_sample_id':'CancID'})
    ge=ge.set_index('CancID')
    ge.to_csv(os.path.join(Path(params["ml_data_outdir"]),'gene_expression.csv'))

    ## Download model specific files
    fname='Data_MCA.zip'
    origin=params['data_url']
    candle.file_utils.get_file(fname, origin)
    # Move model-specific files to ml_data_outdir
    shutil.copy(os.path.join(os.environ['CANDLE_DATA_DIR'],'common','Data','2128_genes.pkl'),os.path.join(Path(params["ml_data_outdir"]),'2128_genes.pkl') )
    shutil.copy(os.path.join(os.environ['CANDLE_DATA_DIR'],'common','Data','smiles_language_chembl_gdsc_ccle.pkl'),os.path.join(Path(params["ml_data_outdir"]),'smiles_language_chembl_gdsc_ccle.pkl') )


def main():
    params = frm.initialize_parameters(
        file_path,
        default_model="Paccmann_MCA_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    run(params)
    print("\nFinished Paccmann MCA pre-processing (transformed raw DRP data to model input ML data).")

if __name__ == "__main__":
    main()
