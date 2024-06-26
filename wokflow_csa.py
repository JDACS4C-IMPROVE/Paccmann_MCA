import parsl
from parsl import python_app , bash_app

from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher # USE the MPIExecLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints

from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.config import Config



from IMPROVE.CLI import CLI
from parsl_apps import preprocess, train, infer
#from IMPROVE.Config.Parsl import Config as Parsl
import IMPROVE.Config.CSA as CSA
from IMPROVE.Config.Common import Config as Common_config
from IMPROVE import framework as frm
import os
from pathlib import Path
import logging
import sys


def build_split_fname(source_data_name, split, phase):
    """ Build split file name. If file does not exist continue """
    if split=='all':
        return f"{source_data_name}_{split}.txt"
    return f"{source_data_name}_split_{split}_{phase}.txt"

# Adjust your user-specific options here:
run_dir="~/tmp"
print(parsl.__version__)

user_opts = {
    "worker_init":      f"source ~/.venv/parsl/bin/activate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE",
    "queue":            "R1819593",
    "walltime":         "1:00:00",
    "nodes_per_block":  10, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
}

additional_definitions = CSA.additional_definitions
#Load CLI parameters
cli = CLI()
cli.set_command_line_options(options=additional_definitions)
cli.get_command_line_options()


## Should we combine csa config and parsl config and use just one initialize_parameter??
common_cfg  = Common_config()
params_csa=common_cfg.initialize_parameters(
                              cli=cli, # Command Line Interface of type CLI
                              section='Global_Params',
                              config_file=cli.params['csa_config_file'],
                              additional_definitions=None,
                              required=None,)
common_cfg  = Common_config()
params_parsl=common_cfg.initialize_parameters(
                              cli=cli, # Command Line Interface of type CLI
                              section='Global_Params',
                              config_file=cli.params['parsl_config_file'],
                              additional_definitions=None,
                              required=None,)

params = {}
params.update(params_csa['Global_Params'])
params.update(params_parsl['Global_Params'])

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']

# Check that environment variable "IMPROVE_DATA_DIR" has been specified - MOVE to initialize_parameters() ???
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

logger = logging.getLogger(f"{params['model_name']}")

params = frm.build_paths(params)  # paths to raw data
maindir = Path(os.environ['IMPROVE_DATA_DIR'])
params['input_dir'] = maindir/params['input_dir']  ### May be add to frm.build_paths()??
params['model_outdir'] = maindir/params['model_outdir']
params['infer_outdir'] = maindir/params['infer_outdir']

""" config = Config(
        executors=[
            HighThroughputExecutor(
                label=params['label'],
                worker_debug=bool(params['worker_debug']),
                cores_per_worker=int(params['cores_per_worker']),
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=int(params['init_blocks']),
                    max_blocks=int(params['max_blocks'])
                )
                #,max_workers_per_node=parsl_config['max_workers_per_node'],
            )
        ],
        strategy=None,
    ) """

#Initialize preprocess futures
preprocess_futures = {key: None for key in params['source_datasets']}
#Initialize train futures
train_futures = {key: {} for key in params['source_datasets']}
#for source in params['source_datasets']:
#    train_futures[source] = {key: None for key in params['split']}

##################### START PARSL PARALLEL EXECUTION #####################

parsl.load()
for source_data_name in params['source_datasets']:
    preprocess_futures[source_data_name] = preprocess(params, source_data_name)  ## MODIFY TO INCLUDE SPLITS IN PARALLEL?

for source_data_name in params['source_datasets']:
    if preprocess_futures[source_data_name].done():
        for split in params['split']:
            train_futures[source_data_name][split] = train(params, source_data_name, split) 

for source_data_name in params['source_datasets']:
    for split in params['split']:
        #if train_futures[source_data_name][split].done():
        for target_data_name in params['target_datasets']:
            infer_futures = infer(params, source_data_name, target_data_name, split, train_futures[source_data_name][split].result())
parsl.clear()