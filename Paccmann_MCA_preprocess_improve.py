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
from improve import framework as frm
from improve import drug_resp_pred as drp
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
import joblib

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

def scale_df(df, scaler_name: str="std", scaler=None, verbose: bool=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        df: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = df.select_dtypes(include="number")

    if scaler is None: # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler ({scaler_name}) is not implemented (no df scaling).")
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm
    return df, scaler

def run(params):
    params = frm.build_paths(params)  # paths to raw data
    processed_outdir = frm.create_ml_data_outpath(params)

    print("\nLoading omics data...")
    oo = drp.OmicsLoader(params)
    print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get the needed canc x data
    ge.index.name = 'CancID' #Model specific modification

    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    print(dd)
    sm = dd.dfs['drug_SMILES.tsv']  # get the needed drug x data
    # Modify files to be compatible with Paccmann_MCA (Model specific modification)
    sm_new = pd.DataFrame(columns = ['SMILES', 'DrugID'])
    sm_new['SMILES'] = sm['canSMILES'].values
    sm_new['DrugID'] = sm.index.values
    
    if not os.path.exists(processed_outdir):
        os.makedirs(processed_outdir, exist_ok=True)
    
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    scaler = None
    for stage, split_file in stages.items():

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

        # Scale features using training data
        if stage == "train":
            # Scale data
            df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
            # Store scaler object
            if params["scaling"] is not None and params["scaling"] != "none":
                scaler_fpath = processed_outdir / params["scaler_fname"]
                joblib.dump(scaler, scaler_fpath)
                print("Scaler object created and stored in: ", scaler_fpath)
        else:
            # Use passed scikit scaler object
            df_canc, _ = scale_df(df_canc, scaler=scaler)

        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        data_fname = frm.build_ml_data_name(params, stage,'.csv')  # e.g., data_fname = train_data.pt
        y_data_fname = f"{stage}_{params['y_data_suffix']}.csv"
        df_y.to_csv(processed_outdir / y_data_fname, index=False)

        # Save SMILES as .smi format as required by the model (Model specific)
        sm_new.to_csv(processed_outdir / "smiles.csv", index=False)
        newfile = os.path.join(file_path,processed_outdir,'smiles.smi')
        file = os.path.join(file_path,processed_outdir,'smiles.csv')
        with open(file,'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  ## skip one line (the first one)
            for line in csv_reader:
                with open(newfile, 'a') as new_txt:    #new file has .txt extn
                    txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
                    txt_writer.writerow(line)   #write the lines to file`

        # Save Gene expression
        ge.to_csv(os.path.join(file_path,processed_outdir,'gene_expression.csv'))

def main():
    params = frm.initialize_parameters(
        file_path,
        default_model="Paccmann_MCA_default_model_csa.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished Paccmann MCA pre-processing (transformed raw DRP data to model input ML data).")

if __name__ == "__main__":
    main()
