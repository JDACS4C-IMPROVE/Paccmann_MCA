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


from IMPROVE.CLI import CLI
from IMPROVE.parsl_apps import preprocess#, train, infer
#from IMPROVE.Config.Parsl import Config as Parsl
import IMPROVE.Config.CSA as CSA
from IMPROVE.Config.Common import Config as Common_config

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
params_cli = cli.params
user_specified_cli=cli.user_specified

"""common_cfg  = Common_config()
params_model=common_cfg.initialize_parameters(
                              cli=cli, # Command Line Interface of type CLI
                              section='Global_Params',
                              config_file=cli.params['model_config_file'],
                              additional_definitions=None,
                              required=None,) """
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
#params.update(params_model['Global_Params'])
params.update(params_csa['Global_Params'])
params.update(params_parsl['Global_Params'])

print(params)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']

# Check that environment variable "IMPROVE_DATA_DIR" has been specified - MOVE to initialize_parameters() ???
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

logger = logging.getLogger(f"{params['model_name']}")

maindir = Path(os.environ['IMPROVE_DATA_DIR'])
params['raw_datadir'] = maindir/params["csa_data_dir"]/ params["raw_data_dir"]
params['x_datadir'] = params['raw_datadir'] / params["x_data_dir"]
params['y_datadir'] = params['raw_datadir'] / params["y_data_dir"]
params['splits_dir'] = params['raw_datadir'] / params["splits_dir"]
params['input_dir'] = maindir/params['input_dir']
params['output_dir'] = maindir/params['output_dir']
print(params['splits_dir'])
print(os.path.exists(params['splits_dir']))


#Implement Preprocess outside Parsl 
preprocess(params)

config = Config(
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
    )

#Run preprocess first
preprocess(params)

futures = {}
parsl.clear()
# checkpoints = get_all_checkpoints(run_dir)
# print("Found the following checkpoints: ", checkpoints)
parsl.load(config)



preprocess_futures = [workflow_csa.preprocess(splits_dir/build_split_fname(source_data_name, split), output_data_dir, params_csa) 
                      for source_data_name in params_csa['source_data_name'] for split in params_csa['split']]

results=workflow_csa.preprocess(params_csa)
#results = Demo.run(config={},debug=True)

for key in results.keys():
    print(f"{key} : {results[key]}")


parsl.clear()