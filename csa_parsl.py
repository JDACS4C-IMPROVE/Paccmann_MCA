""" Python implementation of cross-study analysis workflow """
# cuda_name = "cuda:6"
cuda_name = "cuda:7"

import os
import subprocess
import warnings
from time import time
from pathlib import Path

import pandas as pd

# IMPROVE/CANDLE imports
from improve import framework as frm

# Parsl imports
import parsl
from parsl import python_app, bash_app, join_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor




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
    return f"{source_data_name}_split_{split}_{phase}.txt"


#@python_app  ## May be implemented separately outside this script or does not need parallelization
def preprocess(source_datasets, split_nums, target_datasets): # 
    for source_data_name in source_datasets:
        # Get the split file paths
        # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
        if len(split_nums) == 0:
            # Get all splits
            split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
            split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
            split_nums = sorted(set(split_nums))
            # num_splits = 1
        else:
            # Use the specified splits
            split_files = []
            for s in split_nums:
                split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))
        files_joined = [str(s) for s in split_files]

        for split in split_nums:
            print_fn(f"Split id {split} out of {len(split_nums)} splits.")
            # Check that train, val, and test are available. Otherwise, continue to the next split.
            # files_joined = [str(s) for s in split_files]
            # TODO: check this!
            for phase in ["train", "val", "test"]:
                fname = build_split_fname(source_data_name, split, phase)
                # print(f"{phase}: {fname}")
                if fname not in "\t".join(files_joined):
                    warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                    continue

            for target_data_name in target_datasets:
                ml_data_dir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"
                if ml_data_dir.exists() is True:
                    continue
                if only_cross_study and (source_data_name == target_data_name):
                    continue # only cross-study
                print_fn(f"\nSource data: {source_data_name}")
                print_fn(f"Target data: {target_data_name}")

                ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"

                if source_data_name == target_data_name:
                    # If source and target are the same, then infer on the test split
                    test_split_file = f"{source_data_name}_split_{split}_test.txt"
                else:
                    # If source and target are different, then infer on the entire target dataset
                    test_split_file = f"{target_data_name}_all.txt"
                
                timer_preprocess = Timer()

                # p1 (none): Preprocess train data
                print_fn("\nPreprocessing")
                train_split_file = f"{source_data_name}_split_{split}_train.txt"
                val_split_file = f"{source_data_name}_split_{split}_val.txt"
                print_fn(f"train_split_file: {train_split_file}")
                print_fn(f"val_split_file:   {val_split_file}")
                print_fn(f"test_split_file:  {test_split_file}")
                print_fn(f"ml_data_outdir:   {ml_data_outdir}")
                preprocess_run = ["python",
                    "Paccmann_MCA_preprocess_improve.py",
                    "--train_split_file", str(train_split_file),
                    "--val_split_file", str(val_split_file),
                    "--test_split_file", str(test_split_file),
                    "--ml_data_outdir", str(ml_data_outdir),
                    "--y_col_name", str(y_col_name)
                ]
                result = subprocess.run(preprocess_run, capture_output=True,
                                        text=True, check=True)
                # print(result.stdout)
                # print(result.stderr)
                timer_preprocess.display_timer(print_fn)

@python_app
def train_infer(source_data_name, split, target_datasets):
    ## All import statements here
    # p2 (p1): Train model
    # Train a single model for a given [source, split] pair
    # Train using train samples and early stop using val samples
    model_outdir = MAIN_MODEL_DIR/f"{source_data_name}"/f"split_{split}"
    for target_data_name in target_datasets:
        ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"  #### We cannot have target data name here ??????
        if model_outdir.exists() is False:
            os.makedirs(os.path.join(model_outdir, 'ckpts'), exist_ok=True) # For storing checkpoints
            train_ml_data_dir = ml_data_outdir
            val_ml_data_dir = ml_data_outdir
            timer_train = Timer()
            print_fn("\nTrain")
            print_fn(f"train_ml_data_dir: {train_ml_data_dir}")
            print_fn(f"val_ml_data_dir:   {val_ml_data_dir}")
            print_fn(f"model_outdir:      {model_outdir}")
            train_run = ["python",
                    "Paccmann_MCA_train_improve.py",
                    "--train_ml_data_dir", str(train_ml_data_dir),
                    "--val_ml_data_dir", str(val_ml_data_dir),
                    "--ml_data_outdir", str(ml_data_outdir),
                    "--model_outdir", str(model_outdir),
                    "--epochs", str(epochs),
                    "--y_col_name", y_col_name,
                    "--ckpt_directory", os.path.join(model_outdir, 'ckpts')
            ]
            result = subprocess.run(train_run, capture_output=True,
                                    text=True, check=True)

            timer_train.display_timer(print_fn)

            #Inference
            test_ml_data_dir = ml_data_outdir
            model_dir = model_outdir
            infer_outdir = MAIN_INFER_OUTDIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            timer_infer = Timer()

            print_fn("\nInfer")
            print_fn(f"test_ml_data_dir: {test_ml_data_dir}")
            print_fn(f"infer_outdir:     {infer_outdir}")
            infer_run = ["python",
                  "Paccmann_MCA_infer_improve.py",
                  "--test_ml_data_dir", str(test_ml_data_dir),
                  "--model_dir", str(model_dir),
                  "--infer_outdir", str(infer_outdir),
                  "--y_col_name", y_col_name,
                  "--model_outdir", str(model_outdir),
                  "--ml_data_outdir", str(ml_data_outdir)
            ]
            result = subprocess.run(infer_run, capture_output=True,
                                    text=True, check=True)
            timer_infer.display_timer(print_fn)
        return True
    
@python_app
def infer_targets(source_data_name, split, target_data_name): ## NOT USED
    infer_outdir = MAIN_INFER_OUTDIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"  #### We cannot have target data name here ??????
    model_outdir = MAIN_MODEL_DIR/f"{source_data_name}"/f"split_{split}"
    test_ml_data_dir = ml_data_outdir
    model_dir = model_outdir
    print_fn("\nInfer")
    print_fn(f"test_ml_data_dir: {test_ml_data_dir}")
    print_fn(f"infer_outdir:     {infer_outdir}")
    infer_run = ["python",
            "Paccmann_MCA_infer_improve.py",
            "--test_ml_data_dir", str(test_ml_data_dir),
            "--model_dir", str(model_dir),
            "--infer_outdir", str(infer_outdir),
            "--y_col_name", y_col_name,
            "--model_outdir", str(model_outdir),
            "--ml_data_outdir", str(ml_data_outdir)
    ]
    result = subprocess.run(infer_run, capture_output=True,
                            text=True, check=True)

@join_app
def infer(source_data_name, split): ### Nested parsl????   ## NOT USED
    timer_infer = Timer()
    for target_data_name in target_datasets:
        if only_cross_study and (source_data_name == target_data_name):
            continue # only cross-study
        infer_targets(source_data_name, split, target_data_name)
    timer_infer.display_timer(print_fn)
    return True



fdir = Path(__file__).resolve().parent
y_col_name = "auc"

maindir = Path(f"./{y_col_name}")
MAIN_ML_DATA_DIR = Path(f"./{maindir}/ml.data")
MAIN_MODEL_DIR = Path(f"./{maindir}/models")
MAIN_INFER_OUTDIR = Path(f"./{maindir}/infer")

# Check that environment variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

params = frm.initialize_parameters(
    fdir,
    default_model="Paccmann_MCA_default_model_csa.txt",
)
main_datadir = Path(os.environ["IMPROVE_DATA_DIR"])
raw_datadir = main_datadir / params["raw_data_dir"]
x_datadir = raw_datadir / params["x_data_dir"]
y_datadir = raw_datadir / params["y_data_dir"]
splits_dir = raw_datadir / params["splits_dir"]

# lg = Logger(main_datadir/"csa.log")
print_fn = print
# print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")

### Source and target data sources
## Set 1 - full analysis
#source_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
source_datasets = ["gCSI"]
target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]

only_cross_study = False

## Splits
split_nums = [0,1]  # all splits
#split_nums = [1,2,3]

## Parameters of the experiment/run/workflow
epochs = 1
# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================
timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
print_fn(f"\nsource_datasets: {source_datasets}")
print_fn(f"target_datasets: {target_datasets}")
print_fn(f"split_nums:      {split_nums}")


# Local Config
local_config = Config(
    executors=[
        HighThroughputExecutor(
            label="hpo_local",
            max_workers=4,
            worker_debug=True,
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            )
            #,max_workers_per_node=10,
        )
    ],
    strategy=None,
)
preprocess(source_datasets, split_nums, target_datasets) #Preprocessing
parsl.load()
train_futures = [train_infer(source_data_name, split, target_datasets) for source_data_name in source_datasets for split in split_nums]
#print(train_futures.done())
print(train_futures[0].result())

# Does Parsl wait for training to finish? 
# Inference need models to finish training 
# Waiting for all models to finish training to start inference is not ideal. DO we know which processes are finished so we can 
# start inference on those models, while waiting for others to finish?
#if train_futures.done():
#   infer_futures = [infer(source_data_name, split) for source_data_name in source_datasets for split in split_nums]

'''''
with parsl.load():
    # Submit the function in parallel
    train_futures = [train(source_data_name, split) for source_data_name in source_datasets for split in split_nums]
    # Does Parsl wait for training to finish? 
    # Inference need models to finish training 
    # Waiting for all models to finish training to start inference is not ideal. DO we know which processes are finished so we can 
    # start inference on those models, while waiting for others to finish?
    if train_futures.done():
        infer_futures = [infer(source_data_name, split) for source_data_name in source_datasets for split in split_nums]

'''''


