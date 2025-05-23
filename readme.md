# DiffCkt Open Source Circuit Dataset README

## 1. Dataset Overview

This Dataset is the official dataset accompanying the paper *"DiffCkt: A Diffusion Model-Based Hybrid Neural Network Framework for Automatic Transistor-Level Generation of Analog Circuits"*. Designed to advance automated analog circuit design, this dataset provides high-quality training and research resources for the field. Constructed based on the TSMC 65nm CMOS process, it contains **over 400,000 pairs of amplifier structures and performance metrics** .

For a detailed illustration of the dataset and how it is built, please refer to

> Diffckt_Appendix.pdf

**A detailed illustration of how the relative performance is calculated is also shown in this file!**

# Dataset Loading Instructions

To use the dataset provided with this work, please follow these steps:

## 1. Dataset Preparation

Download the `data_chunks_1.zip` ~ `data_chunks_12.zip` files as well as the `unzip_and_load.py` script to your local working directory (e.g., `/mypath`)

## 2. Data Processing

Execute the processing script:

```shell
python unzip_and_load.py
```

The extracted data will be stored in the `/mypath/unzipped_data` directory.

## 3. Loading the Dataset

The `unzip_and_load.py` script provides a `load_data()` function that returns a `torch.utils.data.Dataset` object containing all circuit samples:

```python
from unzip_and_load import load_data 

# Customize the data path if needed (default: 'unzipped_data')
dir_path = 'unzipped_data'  
circuit_dataset = load_data(dir_path=dir_path)
```

## Dataset Specifications

- **Total samples**: 456,872 circuits
- **Sample structure** (see Appendix A/B of our paper for details):
  - `X: torch.Size([22, 21])` (Node features)
  - `E: torch.Size([22, 22, 25])` (Edge features)
  - `Y: torch.Size([13])` (Circuit performances)
  - `node_mask: torch.Size([22])` (Node mask)
