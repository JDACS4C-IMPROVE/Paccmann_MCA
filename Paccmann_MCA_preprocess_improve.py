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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path
from urllib.request import urlretrieve


# [Req] IMPROVE imports
# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

from model_params_def import preprocess_params # [Req]

filepath = Path(__file__).resolve().parent # [Req]



def get_file(fname, origin, data_dir):
    fpath = os.path.join(data_dir, fname)
    urlretrieve(origin, fpath)
    shutil.unpack_archive(fpath, data_dir)


def run(params):
    print("\nLoads omics data.")
    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

    print("\nLoad drugs data.")
    drugs_obj = drugs_utils.DrugsLoader(params)
    sm = drugs_obj.dfs['drug_SMILES.tsv']  # return SMILES data


    # Modify files to be compatible with Paccmann_MCA (Model specific modification)
    sm_new = pd.DataFrame(columns = ['SMILES', 'DrugID'])
    sm_new['SMILES'] = sm['canSMILES'].values
    sm_new['DrugID'] = sm.index.values
    
    sm = sm.reset_index()
    sm.columns = [params["drug_col_name"], "SMILES"]

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():

        # ---------------------------------
        # [Req] Load response data
        # ------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]
        
        
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        rsp = rsp.merge( ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(sm[params["drug_col_name"]], on=params["drug_col_name"], how="inner")

        ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
        smi_sub = sm_new[sm_new['DrugID'].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

        # Sub-select desired response column (y_col_name)
        # ... and reduce response df to 3 columns: drug_id, cell_id and selected drug_response
        rsp_cut = rsp[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]].copy()

        print(rsp_cut[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        rsp_cut = rsp_cut[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        rsp_cut = rsp_cut.rename(columns = {params["drug_col_name"]:'drug', params["canc_col_name"]:'cell_line'}) # Model specfic change in column names
        rsp_cut[params["y_col_name"]] = rsp_cut['auc']
        rsp_cut.reset_index(inplace=True)

        # [Req] Create data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        frm.save_stage_ydf(ydf=rsp, stage=stage, output_dir=params["output_dir"])

    # Save SMILES as .smi format as required by the model (Model specific)
    if os.path.exists(os.path.join(Path(params["output_dir"]),'smiles.smi')): # Remove smiles.smi to prevent duplicates
        os.remove(os.path.join(Path(params["output_dir"]),'smiles.smi'))
    smi_sub.to_csv(Path(params["output_dir"]) / "smiles.csv", index=False)
    newfile = os.path.join(Path(params["output_dir"]),'smiles.smi')
    file = os.path.join(Path(params["output_dir"]),'smiles.csv')
    with open(file,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  ## skip one line (the first one)
        for line in csv_reader:
            with open(newfile, 'a') as new_txt:    #new file has .txt extn
                txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
                txt_writer.writerow(line)   #write the lines to file`
    # Save Gene expression
    ge_sub=ge_sub.rename(columns={'improve_sample_id':'CancID'})
    ge_sub=ge_sub.set_index('CancID')
    ge_sub.to_csv(os.path.join(Path(params["output_dir"]),'gene_expression.csv'))

    ## Download model specific files
    fname='Data_MCA.zip'
    origin=params['data_url']
    if not os.path.exists(params["input_supp_data_dir"]):
        os.makedirs(params["input_supp_data_dir"], exist_ok=True)
        get_file(fname, origin, params["input_supp_data_dir"])

    # Move model-specific files to output_dir
    #shutil.copy(os.path.join(params["input_supp_data_dir"],'Data','2128_genes.pkl'),os.path.join(Path(params["output_dir"]),'2128_genes.pkl') )
    #shutil.copy(os.path.join(os.environ['CANDLE_DATA_DIR'],'common','Data','smiles_language_chembl_gdsc_ccle.pkl'),os.path.join(Path(params["output_dir"]),'smiles_language_chembl_gdsc_ccle.pkl') )

def main():
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="Paccmann_MCA_default_model_csa.txt",
        additional_definitions=additional_definitions
    )
    run(params)
    print("\nFinished Paccmann MCA pre-processing (transformed raw DRP data to model input ML data).")

if __name__ == "__main__":
    main()
