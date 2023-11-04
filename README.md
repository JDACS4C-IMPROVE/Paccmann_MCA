[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml)

# PACCMANN PREDICTOR

Drug interaction prediction with PaccMann.

`paccmann_predictor` is a package for drug interaction prediction, with examples of 
anticancer drug sensitivity prediction and drug target affinity prediction. Please see the paper:

- [_Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders_](https://doi.org/10.1021/acs.molpharmaceut.9b00520) (*Molecular Pharmaceutics*, 2019). This is the original paper on IC50 prediction using drug properties and tissue-specific cell line information (gene expression profiles). While the original code was written in `tensorflow` and is available [here](https://github.com/drugilsberg/paccmann), this is the `pytorch` implementation of the best PaccMann architecture (multiscale convolutional encoder).


*NOTE*: PaccMann acronyms "Prediction of AntiCancer Compound sensitivity with Multi-modal Attention-based Neural Networks".


## Requirements

- `conda>=3.7`

## Installation

This model is curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE)
The original code is [_here_](https://github.com/PaccMann/paccmann_predictor)


Create a conda environment:

```sh
conda env create -f examples/IC50/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_predictor
```

Install in editable mode for development:

```sh
pip install -e .
```
Install CANDLE package
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```
# Using Default Data

## Example usage with conda environment

**Preprocess (optional)**
```sh
bash preprocess.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```
**Training**
```sh
bash train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```
**Testing**
```sh
bash infer.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```

## Example usage with singularity container
Model definition file 'Paccmann_MCA.def' is located [_here_](https://github.com/JDACS4C-IMPROVE/Singularity/tree/develop/definitions) 

Build Singularity 
```sh
singularity build --fakeroot Paccmann_MCA.sif Paccmann_MCA.def 
```

Execute within container
```sh
singularity exec --nv Paccmann_MCA.sif train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```
# Using Benchmark Data for Cross-Study Analysis
Clone the develop branch of this repository.
Directory structure to use the IMPROVE benchmark data 
```
mkdir csa_data
mkdir csa_data/raw_data
mkdir csa_data/raw_data/y_data
mkdir csa_data/raw_data/x_data
mkdir csa_data/raw_data/splits
mkdir candle_data_dir
mkdir candle_data_dir/CSA_data
```
Follow these steps to download data from the IMPROVE FTP:
```
wget -P csa_data/raw_data/y_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/y_data/response.tsv

wget -P csa_data/raw_data/x_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/x_data/cancer_gene_expression.tsv

wget -P csa_data/raw_data/x_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/x_data/drug_SMILES.tsv
```

To train and test on CCLE split 0, download these files from the IMPROVE FTP:
```
wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_train.txt

wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_val.txt

wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_test.txt
```
## To run the models:
**Preprocess**
```sh
bash preprocess_csa.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR $SPLIT $TRAIN_SOURCE $TEST_SOURCE
```
for example (split=0, train source=CCLE, test source=CCLE):
```sh
bash preprocess_csa.sh 1 candle_data_dir 0 CCLE CCLE
```
**Training**
```sh
bash train_csa.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```
for example:
```sh
bash train_csa.sh 1 candle_data_dir
```
**Testing**
```sh
bash infer_csa.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```
for example:
```sh
bash infer_csa.sh 1 candle_data_dir
```


## References

If you use `paccmann_predictor` in your projects, please cite the following:

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
