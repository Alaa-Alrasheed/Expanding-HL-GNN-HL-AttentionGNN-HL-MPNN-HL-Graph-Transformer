import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import degree
import time
import numpy as np
import torch.nn.functional as F

# IMPORT UNIFIED MODULES
from models import *
from utils import *


def count_parameters(model):
    """Counts the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pe(data):
    """Calculates Degree Positional Encoding"""
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).float()
    pe = torch.log1p(deg).unsqueeze(-1)
    pe = (pe - pe.mean()) / (pe.std() + 1e-12)
    return pe


def create_eval_loaders(data, pos_edges, neg_edges, batch_size, num_neighbors):
    loader_pos = LinkNeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        edge_label_index=pos_edges, edge_label=torch.ones(pos_edges.size(1)),
        shuffle=False, num_workers=0
    )
    loader_neg = LinkNeighborLoader(
        data, num_neighbors=num_neighbors, batch_size=batch_size,
        edge_label_index=neg_edges, edge_label=torch.zeros(neg_edges.size(1)),
        shuffle=False, num_workers=0
    )
    return loader_pos, loader_neg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Computers',
                        choices=['Computers', 'Photo', 'Cora', 'Citeseer', 'Pubmed'])
    parser.add_argument('--model', type=str, default='GT', choices=['MPNN', 'Attn', 'GT'])

    # Architecture Args
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=2)

    # MLP Args
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden', type=int, default=256)

    # Training Args
    parser.add_argument('--batch_size', type=int, default=4096)  # Amazon only
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--att_dropout', type=float, default=0.2)  # Attn and GT only
    parser.add_argument('--layer_dropout', type=float, default=0.1)  # Attn only
    parser.add_argument('--drop_path', type=float, default=0.2)  # GT only

    parser.add_argument('--neg_sampling_ratio', type=float, default=1.0)  # Amazon only
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[20, 10])  # Amazon only
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=3)

    # === NEW: BETA ABLATION ARGUMENT ===
    parser.add_argument('--beta_mode', type=str, default='learnable',
                        choices=['learnable', 'uniform', 'exponential'])

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running {args.model} on {args.dataset} | Mode: {args.beta_mode} | Device: {device} | Runs: {args.runs}")

    # ==========================
    # 1. LOAD DATASET (Once)
    # ==========================
    if args.dataset in ['Computers', 'Photo']:
        path = f"data/Amazon"
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
    else:
        path = f"data/Planetoid"
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())

    # Here we fix the split for consistent comparison across seeds
    data_orig, split_edge = do_edge_split(dataset)
    data_orig = data_orig.to(device)
    data_orig.edge_index = split_edge['train']['edge'].t().to(device)

    pe_dim = 0
    if args.model == 'GT':
        data_orig.pe = get_pe(data_orig).to(device)
        pe_dim = 1
    else:
        data_orig.pe = None

    # Loaders for Amazon
    is_planetoid = args.dataset in ['Cora', 'Citeseer', 'Pubmed']
    train_loader, val_loader_pos, val_loader_neg, test_loader_pos, test_loader_neg = None, None, None, None, None

    if not is_planetoid:
        train_loader = LinkNeighborLoader(
            data_orig, num_neighbors=args.num_neighbors, batch_size=args.batch_size,
            edge_label_index=split_edge['train']['edge'].t(),
            edge_label=torch.ones(split_edge['train']['edge'].size(0)),
            neg_sampling_ratio=args.neg_sampling_ratio, shuffle=True, num_workers=0
        )
        val_loader_pos, val_loader_neg = create_eval_loaders(
            data_orig, split_edge['valid']['edge'].t(), split_edge['valid']['edge_neg'].t(),
            args.batch_size, args.num_neighbors
        )
        test_loader_pos, test_loader_neg = create_eval_loaders(
            data_orig, split_edge['test']['edge'].t(), split_edge['test']['edge_neg'].t(),
            args.batch_size, args.num_neighbors
        )

    # Update logger name to include beta mode
    logger = ResultsLogger(dataset_name=args.dataset, model_name=f"{args.model}_{args.beta_mode}")

    # ==========================
    # 2. MULTI-SEED LOOP
    # ==========================
    for run in range(args.runs):
        print(f"\n=== Run {run + 1}/{args.runs} ===")
        set_seed(42 + run)

        run_start_time = time.time()

        # NEW: Store epoch history for this run
        run_history = []

        # Initialize Model (Pass beta_mode)
        if args.model == 'MPNN':
            model = HLMPNN(data_orig.num_features,
                           args.hidden_channels,
                           args.layers,
                           msg_hidden=128,
                           use_activation=False,
                           beta_mode=args.beta_mode).to(device)
        elif args.model == 'Attn':
            model = HLAttention(data_orig.num_features,
                                args.hidden_channels,
                                args.layers,
                                heads=args.heads,
                                att_dropout=args.att_dropout,
                                layer_dropout=args.layer_dropout,
                                use_activation=True,
                                beta_mode=args.beta_mode).to(device)
        elif args.model == 'GT':
            model = HLGT(data_orig.num_features,
                         args.hidden_channels,
                         args.layers,
                         heads=args.heads,
                         att_dropout=args.att_dropout,
                         drop_path=args.drop_path,
                         pe_dim=pe_dim,
                         use_activation=True,
                         beta_mode=args.beta_mode).to(device)

        print(f"Total Trainable Parameters: {count_parameters(model):,}")

        predictor = LinkPredictor(args.hidden_channels, args.mlp_layers, args.mlp_hidden, args.dropout).to(device)

        # Optimization: Only optimize beta if it is 'learnable'
        beta_params = []
        if args.beta_mode == 'learnable':
            beta_params = [p for n, p in model.named_parameters() if 'beta' in n]

        other_params = [p for n, p in model.named_parameters() if 'beta' not in n]
        other_params.extend(list(predictor.parameters()))

        if args.wd > 0:
            param_groups = [{'params': other_params, 'lr': args.lr, 'weight_decay': args.wd}]
            if beta_params:
                param_groups.append({'params': beta_params, 'lr': 0.05, 'weight_decay': 0.0})
            optimizer = torch.optim.AdamW(param_groups)
        else:
            param_groups = [{'params': other_params, 'lr': args.lr}]
            if beta_params:
                param_groups.append({'params': beta_params, 'lr': 0.05})
            optimizer = torch.optim.Adam(param_groups)

        scaler = torch.amp.GradScaler('cuda')

        # Training
        best_val_auc = 0.0
        best_test_auc = 0.0
        best_test_hits = None

        for epoch in range(1, args.epochs + 1):
            start_t = time.time()

            if is_planetoid:
                loss = train_full_batch(model, predictor, data_orig, split_edge, optimizer, device, model_name=args.model)
                val_res = evaluate_planetoid(model, predictor, data_orig, split_edge, device, model_name=args.model)
                test_res = val_res

                if best_test_hits is None or test_res['Hits@100'] > best_test_hits:
                    best_test_hits = test_res['Hits@100']
                    best_test_auc = test_res['AUC']
                    best_val_auc = test_res['AUC']

                # NEW: Append Hits@100 to history
                run_history.append(test_res['Hits@100'])

                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Test AUC: {test_res['AUC']:.4f} | Hits@100: {test_res['Hits@100']:.4f} | "
                      f"Time: {(time.time() - start_t):.2f}s")
            else:
                loss = train_mini_batch(model, predictor, train_loader, optimizer, device, scaler)
                val_res = evaluate_mini_batch(model, predictor, val_loader_pos, val_loader_neg, device)
                test_res = evaluate_mini_batch(model, predictor, test_loader_pos, test_loader_neg, device)

                if val_res['AUC'] > best_val_auc:
                    best_val_auc = val_res['AUC']
                    best_test_auc = test_res['AUC']

                # NEW: Append AUC to history
                run_history.append(test_res['AUC'])

                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Val AUC: {val_res['AUC']:.4f} | Test AUC: {test_res['AUC']:.4f} | "
                      f"Time: {(time.time() - start_t):.2f}s")

        if is_planetoid:
            print(f"Run {run + 1} | Best Test Hits@100: {best_test_hits:.4f} | Time: {(time.time() - run_start_time):.2f}s")
        else:
            print(f"Run {run + 1} | Best Test AUC: {best_test_auc:.4f} | Time: {(time.time() - run_start_time):.2f}s")

        # Extract Beta Weights for Logging (Handle Learnable vs Fixed)
        if hasattr(model, 'beta'):
            if args.beta_mode == 'learnable':
                # Softmax the parameter
                run_betas = F.softmax(model.beta.detach().cpu(), dim=0).numpy().tolist()
            else:
                # Normalize the fixed buffer
                run_betas = (model.beta / model.beta.sum()).detach().cpu().numpy().tolist()
        else:
            run_betas = None

        # NEW: Pass epoch_data (run_history) to the logger
        logger.finish_run(
            best_val=best_val_auc,
            best_test=best_test_auc,
            best_hits=best_test_hits,
            beta_weights=run_betas,
            epoch_data=run_history
        )

    logger.summary()
    logger.save(args)