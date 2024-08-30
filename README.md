[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml)

# Paccmann_MCA
This repository demonstrates how to use the [IMPROVE library v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta) for building a drug response prediction (DRP) model using Paccmann_MCA, and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.0.3-beta`, is the final release before transitioning to `v0.1.0-alpha`, which introduces a new API. Version `v0.0.3-beta` and all previous releases have served as the foundation for developing essential components of the IMPROVE software stack. Subsequent releases build on this legacy with an updated API, designed to encourage broader adoption of IMPROVE and its curated models by the research community.

A more detailed tutorial can be found [here](https://jdacs4c-improve.github.io/docs/v0.0.3-beta/content/ModelContributorGuide.html).

**Drug interaction prediction with PaccMann**

`paccmann_predictor` is a package for drug interaction prediction, with examples of 
anticancer drug sensitivity prediction and drug target affinity prediction. Please see the paper:

- [_Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders_](https://doi.org/10.1021/acs.molpharmaceut.9b00520) (*Molecular Pharmaceutics*, 2019). This is the original paper on IC50 prediction using drug properties and tissue-specific cell line information (gene expression profiles). While the original code was written in `tensorflow` and is available [here](https://github.com/drugilsberg/paccmann), this is the `pytorch` implementation of the best PaccMann architecture (multiscale convolutional encoder).

*NOTE*: PaccMann acronyms "Prediction of AntiCancer Compound sensitivity with Multi-modal Attention-based Neural Networks".
This model is curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE)
The original model code is [_here_](https://github.com/PaccMann/paccmann_predictor)

## Dependencies
Requirements
- `conda>=3.7`

Create a conda environment:

```sh
conda env create -f examples/IC50/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_predictor
pip install -e .
```
Install CANDLE library
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```
ML framework:
+ [Torch](https://pytorch.org/) -- deep learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta)
+ [candle_lib](https://github.com/ECP-CANDLE/candle_lib) - IMPROVE dependency (enables various hyperparameter optimization on HPC machines) 

## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```
## Model scripts and parameter file
+ `Paccmann_MCA_preprocess_improve.py` - takes benchmark data files and transforms into files for training and inference
+ `Paccmann_MCA_train_improve.py` - trains the Paccmann_MCA model
+ `Paccmann_MCA_infer_improve.py` - runs inference with the trained model
+ `Paccmann_MCA_default_model_csa.txt` - default parameter file

# Step-by-step instructions

### 1. Clone the model repository
```
git clone https://github.com/JDACS4C-IMPROVE/Paccmann_MCA.git
cd Paccmann_MCA
git checkout v0.0.3-beta
```
### 2. Activate environment
Activate conda env
```
conda activate paccmann_predictor
```
### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```
This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout tag `v0.0.3-beta`) outside the Paccmann_MCA model repo
3. Set up env variables: `IMPROVE_DATA_DIR` (to `./csa_data/`) and `PYTHONPATH` (adds IMPROVE repo).

### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python Paccmann_MCA_preprocess_improve.py
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`
* gene_expression and drug SMILES data
* downloads and preprocesses model specifile files required for Paccmann_MCA.

```
ml_data
└── GDSCv1-CCLE
    └── split_0
        ├── test_y_data.csv
        ├── train_y_data.csv
        ├── val_y_data.csv
        ├── gene_expression.csv
        ├── smiles.smi
        ├── 2128_genes.pkl
        └── smiles_language_chembl_gdsc_ccle.pkl

```

### 5. Train model
```bash
python Paccmann_MCA_train_improve.py
```

Trains model using the preprocessed model input data.

Generates:
* trained model: `model.pt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── GDSCv1
    └── split_0
        ├── model.pt
        ├── final_params.pickle
        ├── val_scores.json
        └── val_y_data_predicted.csv
```
### 6. Run inference on test data with the trained model
```python Paccmann_MCA_infer_improve.py```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── GDSCv1-CCLE
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```


## References

```bib
@article{manica2019paccmann,
  title={Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders},
  author={Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian, Vigneshwari and S{\'a}ez-Rodr{\'\i}guez, Julio and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  journal={Molecular pharmaceutics},
  volume={16},
  number={12},
  pages={4797--4806},
  year={2019},
  publisher={ACS Publications},
  doi = {10.1021/acs.molpharmaceut.9b00520},
  note = {PMID: 31618586}
}

@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}
```
