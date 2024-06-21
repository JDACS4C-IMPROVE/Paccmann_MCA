import os
import subprocess
import warnings
from time import time
from pathlib import Path
import pandas as pd
import logging
import sys
import numpy as np

# Parsl imports
import parsl
from parsl import python_app, bash_app, join_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from IMPROVE import framework as frm


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = "auc"

logger = logging.getLogger(f'Start workflow')

class Timer:
  """ Measure time. """
  def __init__(self):
    self.start = time()

  def timer_end(self):
    self.end = time()
    return self.end - self.start

  def display_timer(self, print_fn=print):
    time_diff = self.timer_end()
    if time_diff // 3600 > 0:
        print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
    else:
        print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )

def build_split_fname(source_data_name, split, phase):
    """ Build split file name. If file does not exist continue """
    if split=='all':
        return f"{source_data_name}_{split}.txt"
    return f"{source_data_name}_split_{split}_{phase}.txt"

#@python_app  ## May be implemented separately outside this script or does not need parallelization
def preprocess(params): # 
    split_nums=params['split']
    ### SOURCE DATASETS ####
    for source_data_name in params['source_datasets']:
        # Get the split file paths
        # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
        if len(split_nums) == 0:
            # Get all splits
            split_files = list((params['splits_path']).glob(f"{source_data_name}_split_*.txt"))
            split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
            split_nums = sorted(set(split_nums))
            # num_splits = 1
        else:
            # Use the specified splits
            split_files = []
            for s in split_nums:
                split_files.extend(list((params['splits_path']).glob(f"{source_data_name}_split_{s}_*.txt")))
        files_joined = [str(s) for s in split_files]
        for split in split_nums:
            print(f"Split id {split} out of {len(split_nums)} splits.")
            # Check that train, val, and test are available. Otherwise, continue to the next split.
            # files_joined = [str(s) for s in split_files]
            # TODO: check this!
            for phase in ["train", "val", "test"]:
                fname = build_split_fname(source_data_name, split, phase)
                # print(f"{phase}: {fname}")
                if fname not in "\t".join(files_joined):
                    warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                    continue
            params['ml_data_outdir'] = params['input_dir']/f"{source_data_name}"/f"split_{split}"
            #if params['ml_data_outdir'].exists() is True:
            #    continue
            frm.create_outdir(outdir=params["ml_data_outdir"])
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            test_split_file = f"{source_data_name}_split_{split}_test.txt"
            print(f"train_split_file: {train_split_file}")
            print(f"val_split_file:   {val_split_file}")
            print(f"test_split_file:  {test_split_file}")
            print(f"ml_data_outdir:   {params['ml_data_outdir']}")
            preprocess_run = ["python",
                "Paccmann_MCA_preprocess_improve.py",
                 "--x_data_path", str(params['x_data_path']),
                 "--y_data_path", str(params['y_data_path']),
                 "--splits_path", str(params['splits_path']),
                "--train_split_file", str(train_split_file),
                "--val_split_file", str(val_split_file),
                "--test_split_file", str(test_split_file),
                "--ml_data_outdir", str(params['ml_data_outdir']),
                "--y_col_name", str(y_col_name)
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)

    ### TARGET DATASETS ####
    for target_data_name in params['target_datasets']:
        test_split_file = f"{target_data_name}_all.txt"
        params['ml_data_outdir'] = params['input_dir']/f"{target_data_name}"/f"split_all"
        timer_preprocess = Timer()
        frm.create_outdir(outdir=params["ml_data_outdir"])
        print(f"test_split_file:  {test_split_file}")
        print(f"ml_data_outdir:   {params['ml_data_outdir']}")
        preprocess_run = ["python",
                "Paccmann_MCA_preprocess_improve.py",
                 "--x_data_path", str(params['x_data_path']),
                 "--y_data_path", str(params['y_data_path']),
                 "--splits_path", str(params['splits_path']),
                "--train_split_file", '',   ## Alternative - Change the default value to ''
                "--val_split_file", '',
                "--test_split_file", str(test_split_file),
                "--ml_data_outdir", str(params['ml_data_outdir']),
                "--y_col_name", str(y_col_name)
            ]
        result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)




        
        
        