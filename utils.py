import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, degree
import random
import numpy as np
import json
import os
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResultsLogger:
    def __init__(self, dataset_name, model_name, metric_name="AUC"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.metric_name = metric_name

        # Per-run storage
        self.run_metrics = []  # List of (best_val_auc, best_test_auc, best_test_hits)
        self.run_betas = []
        self.run_histories = [] # <--- NEW: Stores per-epoch metrics for every run
        self.start_time = time.time()

    def finish_run(self, best_val, best_test, best_hits=None, beta_weights=None, epoch_data=None):
        """Called at the end of a seed run"""
        self.run_metrics.append((best_val, best_test, best_hits))
        if beta_weights is not None:
            self.run_betas.append(beta_weights)
        if epoch_data is not None:
            self.run_histories.append(epoch_data) # <--- NEW: Save history

    def summary(self):
        if len(self.run_metrics) == 0:
            print("No results to summarize.")
            return

        clean_metrics = []
        for m in self.run_metrics:
            hits = m[2] if m[2] is not None else 0.0
            clean_metrics.append([m[0], m[1], hits])

        results = torch.tensor(clean_metrics) * 100

        test_mean = results[:, 1].mean().item()
        test_std = results[:, 1].std().item()
        hits_mean = results[:, 2].mean().item()
        hits_std = results[:, 2].std().item()

        print(f"\n==== Final Summary ({len(self.run_metrics)} Runs) ====")
        print(f"Dataset: {self.dataset_name} | Model: {self.model_name}")
        print(f"Best Test AUC:      {test_mean:.2f} ± {test_std:.2f}")
        if hits_mean > 0:
            print(f"Best Test Hits@100: {hits_mean:.2f} ± {hits_std:.2f}")

    def save(self, args):
        os.makedirs("logs", exist_ok=True)

        clean_metrics = []
        for m in self.run_metrics:
            hits = m[2] if m[2] is not None else 0.0
            clean_metrics.append([m[0], m[1], hits])

        results_tensor = torch.tensor(clean_metrics)
        test_mean = results_tensor[:, 1].mean().item()
        test_std = results_tensor[:, 1].std().item()
        hits_mean = results_tensor[:, 2].mean().item()

        if self.run_betas:
            avg_beta = np.mean(np.array(self.run_betas), axis=0).tolist()
        else:
            avg_beta = []

        run_data = {
            "config": vars(args),
            "results": {
                "test_mean": test_mean,
                "test_std": test_std,
                "hits_mean": hits_mean,
                "raw_runs": self.run_metrics,
                "beta_weights": avg_beta,
                "epoch_history": self.run_histories # <--- NEW: Saved to JSON
            },
            "total_time": time.time() - self.start_time
        }

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"logs/{self.dataset_name}_{self.model_name}_K{args.layers}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(run_data, f, indent=4)
        print(f"\n[Saved] Results saved to: {filename}")


# ... (Rest of utils.py remains exactly the same) ...
def do_edge_split(dataset, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)
    full_edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes

    data = train_test_split_edges(data, val_ratio, test_ratio)

    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    num_neg_samples = full_edge_index.size(1)
    neg_edge_index = negative_sampling(
        full_edge_index, num_nodes=num_nodes,
        num_neg_samples=num_neg_samples)

    n_v = data.val_pos_edge_index.size(1)
    n_t = data.test_pos_edge_index.size(1)
    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return data, split_edge


def hard_negative_sampling(edge_index, num_nodes, num_neg, device):
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes).to(device)
    prob = deg / deg.sum()
    neg_v = torch.multinomial(prob, num_neg, replacement=True)
    neg_u = torch.randint(0, num_nodes, (num_neg,), device=device)
    return torch.stack([neg_u, neg_v], dim=1)


def train_full_batch(model, predictor, data, split_edge, optimizer, device, model_name='MPNN'):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(device)

    if model_name == 'Attn':
        neg_train_edge = hard_negative_sampling(
            data.edge_index, data.num_nodes, pos_train_edge.size(0), device
        ).t().to(device)

        optimizer.zero_grad()
        pe = data.pe.to(device) if hasattr(data, 'pe') and data.pe is not None else None
        h = model(data.x, data.edge_index, pe=pe)

        h = F.normalize(h, p=2, dim=1)
        h = F.dropout(h, p=0.2, training=True)
        tau = 0.5

        pos_logits = predictor(h, pos_train_edge) / tau
        neg_logits = predictor(h, neg_train_edge) / tau

        loss = torch.mean(F.relu(1.0 - pos_logits + neg_logits))

    else:
        neg_train_edge = negative_sampling(
            data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=pos_train_edge.size(0)
        ).t().to(device)

        optimizer.zero_grad()
        pe = data.pe.to(device) if hasattr(data, 'pe') and data.pe is not None else None
        h = model(data.x, data.edge_index, pe=pe)

        pos_logits = predictor(h, pos_train_edge)
        neg_logits = predictor(h, neg_train_edge)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_planetoid(model, predictor, data, split_edge, device, model_name='MPNN', split='test'):
    model.eval()
    predictor.eval()

    pe = data.pe.to(device) if hasattr(data, 'pe') and data.pe is not None else None
    h = model(data.x, data.edge_index, pe=pe)

    pos_edge = split_edge[split]['edge'].to(device)
    neg_edge = split_edge[split]['edge_neg'].to(device)

    if model_name == 'Attn':
        h = F.normalize(h, p=2, dim=1)
        tau = 0.5
        pos_pred = (predictor(h, pos_edge) / tau).sigmoid().cpu()
        neg_pred = (predictor(h, neg_edge) / tau).sigmoid().cpu()
    else:
        pos_pred = predictor(h, pos_edge).sigmoid().cpu()
        neg_pred = predictor(h, neg_edge).sigmoid().cpu()

    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    y_score = torch.cat([pos_pred, neg_pred])
    auc = roc_auc_score(y_true, y_score)

    k = 100
    hits = 0
    neg_pred_tensor = neg_pred.view(1, -1)

    try:
        ranks = (neg_pred_tensor > pos_pred.view(-1, 1)).sum(dim=1) + 1
        hits = (ranks <= k).sum().item()
    except RuntimeError:
        for i in range(pos_pred.size(0)):
            rank = (neg_pred > pos_pred[i]).sum().item() + 1
            if rank <= k:
                hits += 1

    hits_100 = hits / pos_pred.size(0)

    return {'AUC': auc, 'Hits@100': hits_100}


def train_mini_batch(model, predictor, loader, optimizer, device, scaler):
    model.train()
    predictor.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pe = batch.pe if hasattr(batch, 'pe') and batch.pe is not None else None

        with torch.amp.autocast('cuda'):
            h = model(batch.x, batch.edge_index, pe=pe)
            out = predictor(h, batch.edge_label_index).view(-1)
            loss = F.binary_cross_entropy_with_logits(out, batch.edge_label.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss)
        total_samples += 1

    return total_loss / total_samples


@torch.no_grad()
def evaluate_mini_batch(model, predictor, loader_pos, loader_neg, device):
    model.eval()
    predictor.eval()

    pos_preds = []
    for batch in loader_pos:
        batch = batch.to(device)
        pe = batch.pe if hasattr(batch, 'pe') and batch.pe is not None else None
        h = model(batch.x, batch.edge_index, pe=pe)
        pos_preds.append(predictor(h, batch.edge_label_index).sigmoid().cpu())

    neg_preds = []
    for batch in loader_neg:
        batch = batch.to(device)
        pe = batch.pe if hasattr(batch, 'pe') and batch.pe is not None else None
        h = model(batch.x, batch.edge_index, pe=pe)
        neg_preds.append(predictor(h, batch.edge_label_index).sigmoid().cpu())

    pos_pred = torch.cat(pos_preds)
    neg_pred = torch.cat(neg_preds)

    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    y_score = torch.cat([pos_pred, neg_pred])

    auc = roc_auc_score(y_true, y_score)

    return {'AUC': auc}