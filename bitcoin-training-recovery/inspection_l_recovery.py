"""
Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering Detection in Bitcoin
Reproduction based on: Lo et al., Applied Intelligence (2023)
https://doi.org/10.1007/s10489-023-04504-9

Pipeline:
  Stage 1 — DGI (self-supervised) with GIN encoder → node embeddings
  Stage 2 — Random Forest (supervised) with embeddings + raw features
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, DeepGraphInfomax
from torch_geometric.utils import to_undirected
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = None          # set to your local path, or use kagglehub (see main)
EPOCHS       = 300
LR           = 1e-4
HIDDEN_DIM   = 128
N_ESTIMATORS = 100
TRAIN_RATIO  = 34            # first 34 time-steps for training
TEST_RATIO   = 15            # last  15 time-steps for testing
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Config] Device: {DEVICE}")


# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────


def load_elliptic(data_dir: str):
    """
    Load the original Elliptic Bitcoin dataset.

    Features: txId + 166 numerical features
    Edges:    txId1, txId2
    Classes:  txId, class

    Original CSV files are never modified.
    """

    print("[Data] Loading Elliptic CSV files...")

    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    # 1. Load original features (CSV has no header)
    feature_cols = ["txId"] + [f"f{i}" for i in range(1, 167)]

    df_feat = pd.read_csv(
        features_path,
        header=None,
        names=feature_cols
    )

    # 2. Load original edges (CSV has a header)
    df_edge_original = pd.read_csv(edges_path)

    # 3. Load original classes (CSV has a header)
    df_cls = pd.read_csv(classes_path)

    # 4. Validate original schemas
    assert df_feat.columns.tolist() == feature_cols
    assert df_edge_original.columns.tolist() == ["txId1", "txId2"]
    assert df_cls.columns.tolist() == ["txId", "class"]

    print("[Schema] Feature columns:", len(df_feat.columns))
    print("[Schema] Edge columns:", df_edge_original.columns.tolist())
    print("[Schema] Class columns:", df_cls.columns.tolist())

    # 5. Normalize transaction IDs to int64
    df_feat["txId"] = pd.to_numeric(
        df_feat["txId"], errors="raise"
    ).astype("int64")

    df_cls["txId"] = pd.to_numeric(
        df_cls["txId"], errors="raise"
    ).astype("int64")

    df_edge_original["txId1"] = pd.to_numeric(
        df_edge_original["txId1"], errors="raise"
    ).astype("int64")

    df_edge_original["txId2"] = pd.to_numeric(
        df_edge_original["txId2"], errors="raise"
    ).astype("int64")

    # 6. Validate labels before mapping
    df_cls["class"] = df_cls["class"].astype(str).str.strip()

    valid_classes = {"1", "2", "unknown"}
    actual_classes = set(df_cls["class"].unique())

    assert actual_classes.issubset(valid_classes), (
        f"Unexpected labels: {actual_classes - valid_classes}"
    )

    # 1 = illicit, 2 = licit, unknown = unlabeled
    label_mapping = {
        "1": 1,
        "2": 0,
        "unknown": -1
    }

    df_cls["label"] = df_cls["class"].map(label_mapping)

    # 7. Validate IDs before merging
    assert df_feat["txId"].is_unique, "Duplicate feature txId"
    assert df_cls["txId"].is_unique, "Duplicate class txId"

    # 8. Merge using original transaction IDs
    df = df_feat.merge(
        df_cls[["txId", "label"]],
        on="txId",
        how="left",
        validate="one_to_one"
    )

    df["label"] = df["label"].fillna(-1).astype(int)

    # 9. Create internal edge mapping for existing GNN code
    # This does not modify the CSV on disk.
    df_edge = df_edge_original.rename(
        columns={
            "txId1": "src",
            "txId2": "dst"
        }
    )

    # 10. Data summary
    print(f"[Data] Nodes: {len(df):,}")
    print(f"[Data] Edges: {len(df_edge):,}")
    print(f"[Data] Illicit: {(df.label == 1).sum():,}")
    print(f"[Data] Licit: {(df.label == 0).sum():,}")
    print(f"[Data] Unknown: {(df.label == -1).sum():,}")

    print("[Data] Loading and schema validation completed.")

    return df, df_edge



def build_temporal_graphs(df: pd.DataFrame, df_edge: pd.DataFrame):
    """
    Split the dataset into 49 temporal graphs by time-step (feature f1).
    Returns a list of PyG Data objects, one per time-step.
    """
    print("[Data] Building temporal graphs …")

    # Global node index mapping: txId → integer index
    txid_to_idx = {txid: i for i, txid in enumerate(df["txId"].values)}
    feature_cols = [f"f{i}" for i in range(1, 167)]
    
    
    # 1. Extract all 166 input features
    X_all = df[feature_cols].values.astype(np.float32)

    # 2. Identify training timesteps using original f1 values
    ts_all = df["f1"].to_numpy()
    timesteps = sorted(np.unique(ts_all))
        
    # 3. First 34 timesteps are the training period
    train_timesteps = timesteps[:TRAIN_RATIO]
    train_mask = np.isin(ts_all, train_timesteps)

    # 4. Fit scaler exclusively on training-period data
    scaler = StandardScaler()
    scaler.fit(X_all[train_mask])

    # 5. Transform all data using the training scaler
    X_all = scaler.transform(X_all).astype(np.float32)
    
    # 6. Extract node labels
    y_all = df["label"].to_numpy()

    print("[Scaler] Training nodes:", train_mask.sum())
    print("[Scaler] Total nodes:", len(X_all))
    print("[Scaler] Input features:", scaler.n_features_in_)
    print("[Scaler] Fitted samples:", scaler.n_samples_seen_)



    # Build edge index (global)
    valid_src, valid_dst = [], []
    for _, row in df_edge.iterrows():
        s, d = row["src"], row["dst"]
        if s in txid_to_idx and d in txid_to_idx:
            valid_src.append(txid_to_idx[s])
            valid_dst.append(txid_to_idx[d])

    edge_index_global = torch.tensor([valid_src, valid_dst], dtype=torch.long)

    graphs = []
    timesteps = sorted(df["f1"].unique())
    for t in timesteps:
        mask     = (ts_all == t)
        node_idx = np.where(mask)[0]                    # global indices in this TS
        local_map = {g: l for l, g in enumerate(node_idx)}

        # Sub-edges: keep only edges where BOTH endpoints are in this time-step
        e_src = edge_index_global[0].numpy()
        e_dst = edge_index_global[1].numpy()
        edge_mask = np.isin(e_src, node_idx) & np.isin(e_dst, node_idx)

        local_src = np.array([local_map[e_src[i]] for i in np.where(edge_mask)[0]])
        local_dst = np.array([local_map[e_dst[i]] for i in np.where(edge_mask)[0]])

        sub_edge = torch.tensor([local_src, local_dst], dtype=torch.long)
        sub_x    = torch.tensor(X_all[node_idx], dtype=torch.float)
        sub_y    = torch.tensor(y_all[node_idx],  dtype=torch.long)

        data = Data(x=sub_x, edge_index=sub_edge, y=sub_y)
        data.global_idx = node_idx          # keep for reference
        data.timestep   = int(t)
        graphs.append(data)

    print(f"[Data] Created {len(graphs)} temporal graphs.")
    return graphs, scaler


# ─────────────────────────────────────────────
# 2.  GIN ENCODER  (Algorithm 2 in paper)
# ─────────────────────────────────────────────

class GINLayer(nn.Module):
    """One GIN layer: MLP(Linear → BN → ReLU → Linear → ReLU)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.conv = GINConv(mlp, train_eps=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GINEncoder(nn.Module):
    """
    Two-layer GIN encoder as described in the paper.
    input_dim → hidden_dim → hidden_dim
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = GINLayer(input_dim, hidden_dim)
        self.layer2 = GINLayer(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = self.layer1(x, edge_index)
        h = self.layer2(h, edge_index)
        return h


# ─────────────────────────────────────────────
# 3.  DGI COMPONENTS  (Section 3.3 in paper)
# ─────────────────────────────────────────────

def corruption(x, edge_index):
    """
    Corruption function C:  randomly shuffle node features,
    keep the same adjacency (A_G = A_H, X_G ≠ X_H).
    Uses Bernoulli-style row permutation.
    """
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm], edge_index


def summary_fn(z, *args, **kwargs):
    """
    Read-out function R: mean pooling + sigmoid  →  global graph summary s̄
    Equation (4) in the paper.
    """
    return torch.sigmoid(z.mean(dim=0))


# ─────────────────────────────────────────────
# 4.  TRAINING  (Algorithm 1 in paper)
# ─────────────────────────────────────────────

def train_dgi(train_graphs, input_dim: int,
              hidden_dim: int = 128,
              epochs: int = 300,
              lr: float = 1e-4):
    encoder = GINEncoder(input_dim, hidden_dim).to(DEVICE)
    model = DeepGraphInfomax(
        hidden_channels = hidden_dim,
        encoder         = encoder,
        summary         = summary_fn,
        corruption      = corruption,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n[DGI] Training on {len(train_graphs)} graphs, "
          f"{epochs} epochs, lr={lr} …")
    model.train()
    loss_history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for data in train_graphs:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(data.x, data.edge_index)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_graphs)
        loss_history.append(float(avg))
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  |  avg loss: {avg:.4f}")
    print("[DGI] Training complete.")
    encoder.loss_history = loss_history
    return encoder

@torch.no_grad()
def extract_embeddings(encoder, graphs):
    """Use the trained encoder to generate node embeddings for all graphs."""
    encoder.eval()
    all_emb, all_y = [], []
    for data in graphs:
        data = data.to(DEVICE)
        emb = encoder(data.x, data.edge_index)
        all_emb.append(emb.cpu().numpy())
        all_y.append(data.y.cpu().numpy())
    return np.vstack(all_emb), np.concatenate(all_y)


# ─────────────────────────────────────────────
# 5.  RF CLASSIFIER  (Section 4.2 in paper)
# ─────────────────────────────────────────────

def build_rf_input(graphs, embeddings, y,
                   raw_features=None, mode="AF+DNE"):
    """
    Construct RF input by concatenating embeddings with raw features.

    mode options:
        "DNE"       — embeddings only
        "LF+DNE"    — embeddings + first 94 local features
        "AF+DNE"    — embeddings + all 166 features  (best in paper)
    """
    assert mode in ("DNE", "LF+DNE", "AF+DNE"), f"Unknown mode: {mode}"

    if mode == "DNE" or raw_features is None:
        X = embeddings
    else:
        n_local = 94
        if mode == "LF+DNE":
            raw = raw_features[:, :n_local]
        else:   # AF+DNE
            raw = raw_features

        X = np.hstack([embeddings, raw])

    # Drop unknown labels (-1) for supervised training/evaluation
    known = (y != -1)
    return X[known], y[known]


def train_rf(X_train, y_train, n_estimators=100, random_state=42):
    print(f"\n[RF] Training Random Forest  "
          f"(trees={n_estimators}, samples={len(X_train):,}) …")
    clf = RandomForestClassifier(
        n_estimators = n_estimators,
        n_jobs       = -1,
        random_state = random_state,
        class_weight = "balanced",   # handle class imbalance
    )
    clf.fit(X_train, y_train)
    print("[RF] Training complete.")
    return clf


def evaluate(clf, X_test, y_test, mode_name=""):
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]   # P(illicit)

    prec  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec   = recall_score   (y_test, y_pred, pos_label=1, zero_division=0)
    f1    = f1_score       (y_test, y_pred, pos_label=1, zero_division=0)
    auc   = roc_auc_score  (y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n{'═'*55}")
    print(f"  Results — Inspection-L  [{mode_name}]")
    print(f"{'─'*55}")
    print(f"  Illicit Precision : {prec:.3f}   (paper AF+DNE: 0.972)")
    print(f"  Illicit Recall    : {rec:.3f}   (paper AF+DNE: 0.721)")
    print(f"  Illicit F1-score  : {f1:.3f}   (paper AF+DNE: 0.828)")
    print(f"  AUC               : {auc:.3f}   (paper AF+DNE: 0.916)")
    print(f"{'─'*55}")
    print(f"  Confusion Matrix:")
    print(f"    TP={cm[1,1]:5d}  FN={cm[1,0]:5d}")
    print(f"    FP={cm[0,1]:5d}  TN={cm[0,0]:5d}")
    print(f"{'═'*55}\n")

    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_dir: str, mode: str = "AF+DNE"):
    """
    Full Inspection-L pipeline.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the three Elliptic CSV files.
    mode : str
        One of "DNE", "LF+DNE", "AF+DNE".
    """
    # ── Load & build graphs ───────────────────────────────────────────────────
    df, df_edge = load_elliptic(data_dir)
    graphs, _   = build_temporal_graphs(df, df_edge)

    # Paper split: first 34 time-steps → train, last 15 → test
    train_graphs = graphs[:TRAIN_RATIO]
    test_graphs  = graphs[TRAIN_RATIO: TRAIN_RATIO + TEST_RATIO]

    input_dim = graphs[0].x.shape[1]   # 166

    # ── Stage 1: DGI training ─────────────────────────────────────────────────
    encoder = train_dgi(
        train_graphs, input_dim,
        hidden_dim = HIDDEN_DIM,
        epochs     = EPOCHS,
        lr         = LR,
    )

    # Extract embeddings for train / test sets
    train_emb, train_y = extract_embeddings(encoder, train_graphs)
    test_emb,  test_y  = extract_embeddings(encoder, test_graphs)

    # Raw feature arrays aligned with embeddings
    def get_raw(graph_list):
        return np.vstack([g.x.numpy() for g in graph_list])

    train_raw = get_raw(train_graphs)
    test_raw  = get_raw(test_graphs)

    # ── Stage 2: RF classification ────────────────────────────────────────────
    X_train, y_train = build_rf_input(
        train_graphs, train_emb, train_y, train_raw, mode=mode)
    X_test,  y_test  = build_rf_input(
        test_graphs,  test_emb,  test_y,  test_raw,  mode=mode)

    clf = train_rf(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)

    # ── Evaluation ────────────────────────────────────────────────────────────
    results = evaluate(clf, X_test, y_test, mode_name=mode)

    return encoder, clf, results


# ─────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 直接指定本地数据集路径
    data_dir = "/Volumes/T9/document/CYBR7901/gcn"

    # Run all three experimental variants (Table 3 in paper)
    all_results = {}
    for variant in ["DNE", "LF+DNE", "AF+DNE"]:
        print(f"\n{'▓'*60}")
        print(f"  Running variant: {variant}")
        print(f"{'▓'*60}")
        _, _, res = run_pipeline(data_dir, mode=variant)
        all_results[variant] = res

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL SUMMARY  (illicit class)")
    print("─"*60)
    print(f"  {'Variant':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("─"*60)
    for v, r in all_results.items():
        print(f"  {v:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['auc']:>8.3f}")
    print("─"*60)
    print("  Paper (AF+DNE)  0.972      0.721    0.828   0.916")
    print("═"*60)
