import os
import subprocess
import warnings
from time import time
from pathlib import Path
import logging
import sys
from parsl import python_app, bash_app, join_app
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

@python_app  ## May be implemented separately outside this script or does not need parallelization
def preprocess(params, source_data_name): # 
    split_nums=params['split']
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
        # TODO: check this!
        for phase in ["train", "val", "test"]:
            fname = build_split_fname(source_data_name, split, phase)
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in params['target_datasets']:
            ml_data_dir = params['input_dir']/f"{source_data_name}-{target_data_name}"
            if ml_data_dir.exists() is True:
                continue
            if params['only_cross_study'] and (source_data_name == target_data_name):
                continue # only cross-study
            print(f"\nSource data: {source_data_name}")
            print(f"Target data: {target_data_name}")

            params['ml_data_outdir'] = params['input_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            frm.create_outdir(outdir=params["ml_data_outdir"])
            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"
            
            timer_preprocess = Timer()

            # p1 (none): Preprocess train data
            print("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            print(f"train_split_file: {train_split_file}")
            print(f"val_split_file:   {val_split_file}")
            print(f"test_split_file:  {test_split_file}")
            print(f"ml_data_outdir:   {params['ml_data_outdir']}")
            preprocess_run = ["python",
                "preprocess.py",
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
            timer_preprocess.display_timer(print)
    return True


@python_app 
def train(params, source_data_name, split): 
    model_outdir = params['model_outdir']/f"{source_data_name}"/f"split_{split}"
    #frm.create_outdir(outdir=model_outdir)
    #for target_data_name in params['target_datasets']:
    ml_data_outdir = params['input_dir']/f"{source_data_name}-{params['target_datasets'][0]}"/f"split_{split}"  #### We cannot have target data name here ??????
    if model_outdir.exists() is False:
        os.makedirs(os.path.join(model_outdir, 'ckpts'), exist_ok=True) # For storing checkpoints
        train_ml_data_dir = ml_data_outdir
        val_ml_data_dir = ml_data_outdir
        timer_train = Timer()
        print("\nTrain")
        print(f"train_ml_data_dir: {train_ml_data_dir}")
        print(f"val_ml_data_dir:   {val_ml_data_dir}")
        print(f"model_outdir:      {model_outdir}")
        train_run = ["python",
                "train.py",
                "--train_ml_data_dir", str(train_ml_data_dir),
                "--val_ml_data_dir", str(val_ml_data_dir),
                "--ml_data_outdir", str(ml_data_outdir),
                "--model_outdir", str(model_outdir),
                "--epochs", str(params['epochs']),
                "--y_col_name", y_col_name,
                "--ckpt_directory", os.path.join(model_outdir, 'ckpts')
        ]
        result = subprocess.run(train_run, capture_output=True,
                                text=True, check=True)
    return True

@python_app  
def infer(params, source_data_name, target_data_name, split): # 
    #for split in params['split']:
    ml_data_outdir = params['input_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    model_outdir = params['model_outdir']/f"{source_data_name}"/f"split_{split}"
    test_ml_data_dir = ml_data_outdir
    infer_outdir = params['infer_outdir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    timer_infer = Timer()

    print("\nInfer")
    print(f"test_ml_data_dir: {test_ml_data_dir}")
    print(f"infer_outdir:     {infer_outdir}")
    infer_run = ["python",
            "infer.py",
            "--test_ml_data_dir", str(test_ml_data_dir),
            "--model_dir", str(model_outdir),
            "--infer_outdir", str(infer_outdir),
            "--y_col_name", y_col_name,
            "--model_outdir", str(model_outdir),
            "--ml_data_outdir", str(ml_data_outdir)
    ]
    result = subprocess.run(infer_run, capture_output=True,
                            text=True, check=True)
    timer_infer.display_timer(print)
    return True