# Hierarchical Learning Graph Attention Network (HL-Attn)

This repository contains the official PyTorch implementation for the paper:  
**"[Insert Exact Paper Title Here]"**

## Overview
This codebase implements the **Hierarchical Learning (HL)** framework for Graph Neural Networks. The flagship model, **HL-Attn**, utilizes a hierarchical aggregation mechanism across multiple attention layers to alleviate oversmoothing and effectively capture both local and long-range structural dependencies. 

It includes the code to reproduce experiments on standard sparse citation networks (Cora, Citeseer, Pubmed) and dense co-purchase graphs (Amazon Photo, Computers).

## Repository Structure
* `main.py`: The main entry point for running experiments. It handles dataset loading, training loops, evaluation, and logging.
* `models.py`: Contains the PyTorch Geometric implementations of the proposed architectures, including `HLAttention`.
* `utils.py`: Helper functions for metrics, data preprocessing, and training utilities.
* `requirements.txt`: List of required Python packages and dependencies.

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda to manage your environment.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
Create and activate a virtual environment (optional but recommended):

'''Bash
conda create -n hl-attn python=3.9 -y
conda activate hl-attn
Install the dependencies:

'''Bash
pip install -r requirements.txt
(Note: Depending on your hardware, you may need to install the specific versions of torch and torch_geometric that match your CUDA version from their official websites).

Usage
You can run the model using main.py. The script will automatically download the required datasets (via PyTorch Geometric) on the first run.

To run a basic experiment on the Cora dataset:

'''Bash
python main.py --dataset Cora
To reproduce specific results from the paper:
You can pass hyperparameters directly via command-line arguments. For example, to run on Amazon Photo with the paper's exact configuration:

'''Bash
python main.py --dataset Photo --layers 6 --hidden_channels 512 --heads 2 --lr 0.001
(Please refer to the hyperparameter tables in the paper's appendix for the exact configurations for each dataset).

Citation
If you find this code or our paper useful in your research, please consider citing our work:


Contact
For any questions or issues with the code, please open an issue on this repository or contact alaa.m.aref@gmail.com.
