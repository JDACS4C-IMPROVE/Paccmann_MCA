# Model-specific params (Model: Paccmann_MCA)
preprocess_params = [
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
     }
]

train_params = [
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

infer_params = [
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