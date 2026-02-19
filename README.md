Hierarchical Graph Models for Link Prediction

Official PyTorch implementation of three hierarchical graph architectures for link prediction:

HL-MPNN (Hierarchical Message Passing Network)

HL-Attn (Hierarchical Graph Attention Network)

HL-GT (Hierarchical Graph Transformer)

The models are evaluated on standard benchmark datasets using multi-seed experiments with automatic logging.

Installation
1. Create environment
conda create -n hl-gnn python=3.10
conda activate hl-gnn
2. Install dependencies
pip install -r requirements.txt

If torch-geometric installation fails, follow the official PyG installation guide:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Running Experiments
Example: HL-Attn on Amazon Computers
python main.py \
  --dataset Computers \
  --model Attn \
  --layers 6 \
  --hidden_channels 128 \
  --heads 2 \
  --epochs 300 \
  --runs 3
Supported Datasets

Amazon Computers

Amazon Photo

Cora

Citeseer

Pubmed

Models
HL-MPNN

Hierarchical multi-layer message passing architecture built on a custom MPNN layer.

HL-Attn

Hierarchical architecture using graph attention layers.

HL-GT

Hierarchical graph transformer architecture with residual connections and feed-forward blocks.

Output

Results are automatically saved in:

logs/

Each log file includes:

Mean and standard deviation across runs

Per-run metrics

Full experiment configuration

Citation

If you use this repository in your research, please cite our paper:
