import parsl
from parsl import python_app , bash_app
import subprocess

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
from time import time
from typing import Sequence, Tuple, Union
run_dir="~/tmp"
print(parsl.__version__)

@python_app
def hello_python (message):
    return 'Hello %s' % message

user_opts = {
    "worker_init":      f". ~/.bashrc ; conda activate parsl; export PYTHONPATH=$PYTHONPATH:/IMPROVE; export IMPROVE_DATA_DIR=./improve_dir; module use /soft/spack/gcc/0.6.1/install/modulefiles/Core; module load apptainer; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE_Aim1",
    "queue":            "debug-scaling",
    "walltime":         "1:00:00",
    "nodes_per_block":  3,# think of a block as one job on polaris, so to run on the main queues, set this >= 10
}


####### CONFIG FOR POLARIS ######
config_polaris = Config(
            retries=1,  # Allows restarts if jobs are killed by the end of a job
            executors=[
                HighThroughputExecutor(
                    label="htex",
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    max_workers=64,
                    available_accelerators=4,  # Ensures one worker per accelerator
                    address=address_by_interface("bond0"),
                    cpu_affinity="block-reverse",
                    prefetch_capacity=0,  # Increase if you have many more tasks than workers
                    start_method="spawn",
                    provider=PBSProProvider(  # type: ignore[no-untyped-call]
                        launcher=MpiExecLauncher(  # Updates to the mpiexec command
                            bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                        ),
                        account="IMPROVE_Aim1",
                        queue="debug-scaling",
                        # PBS directives (header lines): for array jobs pass '-J' option
                        scheduler_options=user_opts['scheduler_options'],
                        worker_init=user_opts['worker_init'],
                        nodes_per_block=10,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel jobs
                        cpus_per_node=64,
                        walltime="1:00:00",
                    ),
                ),
            ],
            run_dir=str(run_dir),
            strategy='simple',
            app_cache=True,
        ) 

parsl.load(config_polaris)
for i in range(40):
    print(i)
    print(hello_python(f"World {i} (Python)").result())

parsl.clear()
