import os
from pathlib import Path
#import model_specific_config - To import model specific parameters


##TODO Replace CANDLE initialize_parameter()

fdir = Path(__file__).resolve().parent
required = None
additional_definitions = [
    {"name": "input_dir",
     "type": str,
     "default": 'input',
     "help": "Input directory"
    },
    {"name": "model_outdir",
     "type": str,
     "default": 'model',
     "help": "Output directory for trained model and checkpoints"
    },
    {"name": "infer_outdir",
     "type": str,
     "default": 'infer',
     "help": "Output directory for inference results"
    },
    {"name": "parsl_config_file", # workflow
     "default": "parsl_config.ini",
     "type": str,
     "required": True,
     "help": "Config file for Parsl",
    },
    {"name": "csa_config_file", # workflow
     "default": "csa_config.ini",
     "type": str,
     "required": True,
     "help": "Config file for CSA workflow",
    },
    {"name": "model_config_file", # workflow
     "default": "model_config.ini",
     "type": str,
     "required": True,
     "help": "Config file for the model",
    },
    {"name": "model_specific_data", # workflow
     "default": True,
     "type": bool,
     "required": True,
     "help": "Does the model require model specific files?",
    },
    {"name": "model_specific_data_url", # workflow
     "default":'https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/Paccmann_MCA/Data_MCA.zip',
     #"default": "",
     "type": str,
     "required": True,
     "help": "url for downloading author data",
    },
    {"name": "train_split_file", # workflow
     "default": "train_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the train split file",
    },
    {"name": "val_split_file", # workflow
     "default": "val_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the validation split file",
    },
    {"name": "test_split_file", # workflow
     "default": "test_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the test split file",
    },
    {"name": "source_datasets",
     "type": list,
     "default": ['CCLE'],
     "help": "source_datasets for cross study analysis"
    },
    {"name": "target_datasets",
     "type": list,
     "default": ["CCLE", "gCSI"],
     "help": "target_datasets for cross study analysis"
    },
    {"name": "source_data_name",  ### DO WE NEED THIS??
     "type": str,
     "default": 'CCLE',
     "help": "Source dataset name for preprocessing"
    },
    {"name": "split",
     "type": list,
     "default": ['0'],
     "help": "Split number for preprocessing"
    },
    {"name": "only_cross_study",
     "type": bool,
     "default": False,
     "help": "If only cross study analysis is needed"
    },
    {"name": "log_level",
     "type": str,
     "default": os.getenv("IMPROVE_LOG_LEVEL", "WARNING"),
     "help": "Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET"
    },
    {"name": "model_name",
     "type": str,
     "default": 'Paccmann_MCA', ## Change the default to LGBM??
     "help": "Name of the deep learning model"
    },
    {"name": "epochs",
     "type": int,
     "default": 1,
     "help": "Number of epochs"
    },
    {"name": "learning_rate",
     "type": float,
     "default": 0.001,
     "help": "Learning rate"
    },
    {"name": "batch_size",
     "type": int,
     "default": 32,
     "help": "Batch size"
    }
    ]
