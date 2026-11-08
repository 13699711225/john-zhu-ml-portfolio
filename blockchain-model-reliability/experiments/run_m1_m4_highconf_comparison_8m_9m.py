import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GINConv

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path("exp_1a_small_block_window_8m_9m")

TX_FILE = BASE_DIR / "small_window_transfers.csv"
POS_FILE = BASE_DIR / "positive_seed.csv"
NEG_FILE = BASE_DIR / "negative_candidates.csv"
DEFAULT_NEG_FILE = NEG_FILE
BITCOIN_ENCODER_PATH = "bitcoin_encoder.pt"
BITCOIN_CLASSIFIER_PATH = "bitcoin_rf.pkl"
BITCOIN_SCALER_PATH = "bitcoin_scaler.pkl"

OUT_CSV = "m1_m4_highconf_comparison_8m_9m.csv"
OUT_JSON = "m1_m4_highconf_comparison_8m_9m.json"
OUT_PROBS = "m1_m4_highconf_probs_8m_9m.csv"

HIDDEN_DIM = 64
K_NEIGHBORS = 10
RANDOM_STATE = 42
N_REPEATS = 30
N_POS_SIZES = [3, 5, 10, 15]
MAX_NEGATIVES = 5000


# =========================================================
# GIN encoder: must match Bitcoin training
# =========================================================

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
            ),
            train_eps=True,
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = GINLayer(input_dim, hidden_dim)
        self.layer2 = GINLayer(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        return self.layer2(self.layer1(x, edge_index), edge_index)


# =========================================================
# Utility
# =========================================================

def norm(x):
    return str(x).strip().lower()


def safe_div(a, b):
    return float(a / b) if b != 0 else 0.0


def best_threshold_f1(y_true, probs):
    p, r, t = precision_recall_curve(y_true, probs)

    if len(t) == 0:
        return 0.0, 0.5, 0.0, 0.0

    f1s = np.where(
        (p[:-1] + r[:-1]) > 0,
        2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9),
        0,
    )

    i = int(np.argmax(f1s))

    return float(f1s[i]), float(t[i]), float(p[i]), float(r[i])


def compute_ece(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if mask.sum() == 0:
            continue

        avg_pred = float(probs[mask].mean())
        actual_frac = float(y_true[mask].mean())
        weight = mask.sum() / n

        ece += weight * abs(avg_pred - actual_frac)

    return float(ece)


def eval_probs(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    pred_05 = (probs >= 0.5).astype(int)

    f1_05 = f1_score(y_true, pred_05, zero_division=0)
    p_05 = precision_score(y_true, pred_05, zero_division=0)
    r_05 = recall_score(y_true, pred_05, zero_division=0)

    f1_best, best_thr, p_best, r_best = best_threshold_f1(y_true, probs)

    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")

    try:
        auc_rev = roc_auc_score(y_true, 1 - probs)
    except Exception:
        auc_rev = float("nan")

    try:
        brier = brier_score_loss(y_true, probs)
    except Exception:
        brier = float("nan")

    try:
        ece = compute_ece(y_true, probs)
    except Exception:
        ece = float("nan")

    return {
        "auc": float(auc),
        "auc_reversed": float(auc_rev),
        "f1_at_0.5": float(f1_05),
        "precision_at_0.5": float(p_05),
        "recall_at_0.5": float(r_05),
        "f1_best": float(f1_best),
        "best_threshold": float(best_thr),
        "precision_best": float(p_best),
        "recall_best": float(r_best),
        "brier": float(brier),
        "ece": float(ece),
    }


# =========================================================
# Inspection-L style 166-dim feature construction
# =========================================================

def build_inspection_l_features(df, target_addresses):
    addr_list = sorted(target_addresses)
    addr_idx = {a: i for i, a in enumerate(addr_list)}

    sg = df.groupby("from_address")
    rg = df.groupby("to_address")

    def safe(g, a, c, f):
        try:
            return float(f(g.get_group(a)[c]))
        except Exception:
            return 0.0

    features = []

    for addr in addr_list:
        ic = safe(rg, addr, "value_usdt", len)
        oc = safe(sg, addr, "value_usdt", len)
        is_ = safe(rg, addr, "value_usdt", sum)
        os_ = safe(sg, addr, "value_usdt", sum)

        local = [
            ic,
            oc,
            ic + oc,
            is_,
            os_,
            is_ - os_,
            safe(rg, addr, "value_usdt", np.mean),
            safe(sg, addr, "value_usdt", np.mean),
            safe(rg, addr, "value_usdt", np.std),
            safe(sg, addr, "value_usdt", np.std),
            safe(rg, addr, "value_usdt", max),
            safe(sg, addr, "value_usdt", max),
            safe(rg, addr, "value_usdt", min),
            safe(sg, addr, "value_usdt", min),
            safe(rg, addr, "from_address", lambda x: float(x.nunique())),
            safe(sg, addr, "to_address", lambda x: float(x.nunique())),
        ]

        for grp in [sg, rg]:
            try:
                ts = grp.get_group(addr)["timestamp"]
                local += [float(ts.min()), float(ts.max() - ts.min())]
            except Exception:
                local += [0.0, 0.0]

        local += [
            is_ / (os_ + 1e-9),
            (ic + oc) / (sum(local[16:20:2]) + 1e-9),
        ]

        local += [0.0] * (94 - len(local))

        try:
            no = sg.get_group(addr)["to_address"].tolist()
        except Exception:
            no = []

        try:
            ni = rg.get_group(addr)["from_address"].tolist()
        except Exception:
            ni = []

        nbrs = list(set(no + ni))

        def s5(lst):
            if not lst:
                return [0.0] * 5
            a = np.array(lst, dtype=float)
            return [
                float(a.mean()),
                float(a.std()),
                float(a.max()),
                float(a.min()),
                float(len(a)),
            ]

        agg = (
            s5([safe(sg, n, "value_usdt", sum) for n in nbrs])
            + s5([safe(rg, n, "value_usdt", sum) for n in nbrs])
            + [0.0] * (72 - 10)
        )

        x = local + agg

        if len(x) != 166:
            raise RuntimeError(f"Feature dimension error: {len(x)}")

        features.append(x)

    X = np.array(features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return addr_list, addr_idx, X


def build_edge_index(df, addr_idx, X):
    src, dst = [], []

    for _, row in df.iterrows():
        s = row["from_address"]
        d = row["to_address"]

        if s in addr_idx and d in addr_idx:
            src.append(addr_idx[s])
            dst.append(addr_idx[d])

    n = len(addr_idx)

    if len(src) < n * 2:
        A = kneighbors_graph(
            X,
            K_NEIGHBORS,
            mode="connectivity",
            include_self=False,
            n_jobs=-1,
        )
        A = (A + A.T).tocoo()
        src += A.row.tolist()
        dst += A.col.tolist()

    return torch.tensor([src, dst], dtype=torch.long)


# =========================================================
# Behaviour covariates
# =========================================================

def build_behaviour_features(df, addr_list):
    out_groups = df.groupby("from_address")
    in_groups = df.groupby("to_address")

    rows = []

    for addr in addr_list:
        out_df = out_groups.get_group(addr) if addr in out_groups.groups else None
        in_df = in_groups.get_group(addr) if addr in in_groups.groups else None

        out_count = 0 if out_df is None else len(out_df)
        in_count = 0 if in_df is None else len(in_df)
        tx_count = in_count + out_count

        total_out_amount = 0.0 if out_df is None else float(out_df["value_usdt"].sum())
        total_in_amount = 0.0 if in_df is None else float(in_df["value_usdt"].sum())
        total_amount = total_in_amount + total_out_amount

        unique_receivers = 0 if out_df is None else int(out_df["to_address"].nunique())
        unique_senders = 0 if in_df is None else int(in_df["from_address"].nunique())

        counterparties = set()
        if out_df is not None:
            counterparties.update(out_df["to_address"].tolist())
        if in_df is not None:
            counterparties.update(in_df["from_address"].tolist())

        block_set = set()
        if out_df is not None:
            block_set.update(out_df["block_number"].tolist())
        if in_df is not None:
            block_set.update(in_df["block_number"].tolist())

        contract_touch_count = 0

        if out_df is not None and "to_is_contract" in out_df.columns:
            contract_touch_count += int((out_df["to_is_contract"] == 1).sum())

        if in_df is not None and "from_is_contract" in in_df.columns:
            contract_touch_count += int((in_df["from_is_contract"] == 1).sum())

        contract_touch_ratio = safe_div(contract_touch_count, tx_count)
        in_out_ratio = safe_div(in_count, out_count)
        amount_in_out_ratio = safe_div(total_in_amount, total_out_amount)

        rows.append({
            "address": addr,
            "tx_count": tx_count,
            "in_count": in_count,
            "out_count": out_count,
            "unique_counterparties": len(counterparties),
            "active_blocks": len(block_set),
            "total_amount": total_amount,
            "contract_touch_ratio": contract_touch_ratio,
            "in_out_ratio": in_out_ratio,
            "amount_in_out_ratio": amount_in_out_ratio,
            "log_tx_count": np.log1p(tx_count),
            "log_unique_counterparties": np.log1p(len(counterparties)),
            "log_active_blocks": np.log1p(len(block_set)),
            "log_total_amount": np.log1p(total_amount),
        })

    return pd.DataFrame(rows)


def make_model_matrix(raw_probs, behaviour_df, model_name):
    """
    M1: raw_score only
    M2: behaviour only
    M3/M4: raw_score + behaviour
    """
    raw_probs = np.asarray(raw_probs)

    behaviour_cols = [
        "log_tx_count",
        "log_unique_counterparties",
        "log_active_blocks",
        "log_total_amount",
        "contract_touch_ratio",
        "in_out_ratio",
    ]

    B = behaviour_df[behaviour_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if model_name == "M1_raw_only_platt":
        return raw_probs.reshape(-1, 1)

    if model_name == "M2_behaviour_only_logistic":
        return B

    if model_name in [
        "M3_raw_plus_behaviour_platt",
        "M4_weighted_raw_plus_behaviour",
    ]:
        return np.column_stack([raw_probs, B])

    raise ValueError(f"Unknown model_name: {model_name}")


# =========================================================
# Sample weighting for M4
# =========================================================

def make_activity_score(behaviour_df):
    cols = [
        "log_tx_count",
        "log_unique_counterparties",
        "log_active_blocks",
        "log_total_amount",
    ]

    X = behaviour_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    if len(X) <= 1:
        return np.zeros(len(X))

    Z = StandardScaler().fit_transform(X)

    return (
        0.35 * Z[:, 0]
        + 0.30 * Z[:, 1]
        + 0.20 * Z[:, 2]
        + 0.15 * Z[:, 3]
    )


def compute_sample_weights(y_cal, behaviour_cal_df):
    """
    final weight = class weight * activity density-ratio weight

    activity density-ratio:
        P_positive(activity_bucket) / P_negative(activity_bucket)
    """
    df = behaviour_cal_df.copy()
    df["label"] = y_cal
    df["activity_score"] = make_activity_score(df)

    try:
        df["activity_bucket"] = pd.qcut(
            df["activity_score"],
            q=min(5, max(2, len(df) // 5)),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        df["activity_bucket"] = 0

    n = len(df)
    n_pos = max(1, int((df["label"] == 1).sum()))
    n_neg = max(1, int((df["label"] == 0).sum()))

    class_weight_pos = n / (2.0 * n_pos)
    class_weight_neg = n / (2.0 * n_neg)

    pos_dist = df[df["label"] == 1]["activity_bucket"].value_counts(normalize=True)
    neg_dist = df[df["label"] == 0]["activity_bucket"].value_counts(normalize=True)

    weights = []

    for _, row in df.iterrows():
        bucket = row["activity_bucket"]

        p_pos = float(pos_dist.get(bucket, 0.0))
        p_neg = float(neg_dist.get(bucket, 0.0))

        activity_weight = (p_pos + 1e-3) / (p_neg + 1e-3)
        class_weight = class_weight_pos if row["label"] == 1 else class_weight_neg

        weights.append(class_weight * activity_weight)

    weights = np.array(weights, dtype=float)

    # Avoid extreme weights in tiny calibration sets.
    weights = np.clip(weights, 0.1, 20.0)

    return weights


# =========================================================
# Frozen model inference
# =========================================================

def load_frozen_model():
    encoder = GINEncoder(input_dim=166, hidden_dim=HIDDEN_DIM)
    encoder.load_state_dict(torch.load(BITCOIN_ENCODER_PATH, map_location="cpu"))
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    clf = joblib.load(BITCOIN_CLASSIFIER_PATH)
    scaler = joblib.load(BITCOIN_SCALER_PATH)

    return encoder, clf, scaler


@torch.no_grad()
def frozen_inference(encoder, clf, scaler, X_raw, edge_index):
    X = scaler.transform(X_raw).astype(np.float32)
    emb = encoder(torch.tensor(X, dtype=torch.float), edge_index).numpy()
    return clf.predict_proba(np.hstack([X, emb]))[:, 1]


# =========================================================
# Model training/evaluation loop
# =========================================================

def fit_predict_logistic(X_cal, y_cal, X_test, sample_weight=None):
    lr = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )

    if sample_weight is None:
        lr.fit(X_cal, y_cal)
    else:
        lr.fit(X_cal, y_cal, sample_weight=sample_weight)

    probs = lr.predict_proba(X_test)[:, 1]

    return probs, lr.coef_[0].tolist(), float(lr.intercept_[0])


def run_one_model(model_name, raw_probs, behaviour_df, y_true, n_pos):
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    X_all = make_model_matrix(raw_probs, behaviour_df, model_name)

    rows = []

    for seed in range(N_REPEATS):
        rng = np.random.RandomState(seed + 1000)

        if n_pos >= len(pos_idx):
            continue

        cal_pos = rng.choice(pos_idx, n_pos, replace=False)
        test_pos = np.array([i for i in pos_idx if i not in set(cal_pos)])

        if len(test_pos) == 0:
            continue

        cal_neg_n = min(n_pos * 10, len(neg_idx))
        cal_neg = rng.choice(neg_idx, cal_neg_n, replace=False)

        used_cal = set(cal_pos) | set(cal_neg)
        remaining_neg = np.array([i for i in neg_idx if i not in used_cal])

        test_neg_n = min(len(test_pos) * 10, len(remaining_neg))

        if test_neg_n == 0:
            continue

        test_neg = rng.choice(remaining_neg, test_neg_n, replace=False)

        cal_idx = np.concatenate([cal_pos, cal_neg])
        test_idx = np.concatenate([test_pos, test_neg])

        y_cal = y_true[cal_idx]
        y_test = y_true[test_idx]

        sample_weight = None

        if model_name == "M4_weighted_raw_plus_behaviour":
            sample_weight = compute_sample_weights(
                y_cal,
                behaviour_df.iloc[cal_idx].reset_index(drop=True),
            )

        probs_test, coef, intercept = fit_predict_logistic(
            X_all[cal_idx],
            y_cal,
            X_all[test_idx],
            sample_weight=sample_weight,
        )

        result = eval_probs(y_test, probs_test)

        result.update({
            "model": model_name,
            "n_pos": int(n_pos),
            "seed": int(seed),
            "coef": coef,
            "intercept": intercept,
        })

        rows.append(result)

    return rows


def aggregate_rows(rows):
    if not rows:
        return {}

    metrics = [
        "auc",
        "auc_reversed",
        "f1_at_0.5",
        "precision_at_0.5",
        "recall_at_0.5",
        "f1_best",
        "precision_best",
        "recall_best",
        "brier",
        "ece",
    ]

    out = {}

    for m in metrics:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")

    out["n_repeats"] = len(rows)

    return out


def print_comparison_table(summary_df):
    keep_cols = [
        "version",
        "model",
        "input",
        "purpose",
        "n_pos",
        "auc_mean",
        "auc_std",
        "f1_at_0.5_mean",
        "f1_best_mean",
        "f1_best_std",
        "brier_mean",
        "ece_mean",
    ]

    print("\n[Model comparison table]")
    print(summary_df[keep_cols].to_string(index=False))


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neg-file",
        type=str,
        default=str(DEFAULT_NEG_FILE),
        help="Negative candidate CSV file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="activitymatched",
        help="Output tag, e.g. random, highconf, activitymatched"
    )
    args = parser.parse_args()

    neg_file = Path(args.neg_file)

    out_csv = f"m1_m4_{args.tag}_comparison_8m_9m.csv"
    out_json = f"m1_m4_{args.tag}_comparison_8m_9m.json"
    out_probs = f"m1_m4_{args.tag}_probs_8m_9m.csv"

    print("=" * 72)
    print("V0 + M1/M2/M3/M4 model comparison")
    print("=" * 72)

    tx = pd.read_csv(TX_FILE)
    pos = pd.read_csv(POS_FILE)
    neg = pd.read_csv(neg_file)

    tx["from_address"] = tx["from_address"].map(norm)
    tx["to_address"] = tx["to_address"].map(norm)
    tx["token_address"] = tx["token_address"].map(norm)
    tx["value_usdt"] = tx["amount"].astype(float) / 1e6

    pos_set = set(pos["address"].map(norm))
    neg_set_all = set(neg["address"].map(norm)) - pos_set

    rng = np.random.RandomState(RANDOM_STATE)
    neg_list = sorted(neg_set_all)

    if len(neg_list) > MAX_NEGATIVES:
        neg_list = list(rng.choice(neg_list, MAX_NEGATIVES, replace=False))

    target_addresses = sorted(pos_set | set(neg_list))

    tx_sub = tx[
        tx["from_address"].isin(target_addresses)
        | tx["to_address"].isin(target_addresses)
    ].copy()

    print(f"[Input] negative file: {neg_file}")
    print(f"[Labels] positives={len(pos_set):,}")
    print(f"[Labels] negatives_all={len(neg_set_all):,}")
    print(f"[Labels] negatives_used={len(neg_list):,}")
    print(f"[Data] target-touching transfers={len(tx_sub):,}")

    addr_list, addr_idx, X_raw = build_inspection_l_features(
        tx_sub,
        set(target_addresses),
    )

    y_true = np.array([1 if a in pos_set else 0 for a in addr_list], dtype=int)

    edge_index = build_edge_index(tx_sub, addr_idx, X_raw)
    behaviour_df = build_behaviour_features(tx_sub, addr_list)

    encoder, clf, scaler = load_frozen_model()
    raw_probs = frozen_inference(encoder, clf, scaler, X_raw, edge_index)

    prob_out = pd.DataFrame({
        "address": addr_list,
        "label": y_true,
        "raw_prob": raw_probs,
    }).merge(behaviour_df, on="address", how="left")

    prob_out.to_csv(out_probs, index=False)

    # V0 baseline
    v0_eval = eval_probs(y_true, raw_probs)

    print("\n[V0 Frozen raw score]")
    print(
        f"  AUC={v0_eval['auc']:.3f} "
        f"Rev={v0_eval['auc_reversed']:.3f} "
        f"F1@0.5={v0_eval['f1_at_0.5']:.3f} "
        f"F1best={v0_eval['f1_best']:.3f}"
    )

    model_meta = {
        "V0_frozen_raw": {
            "version": "V0",
            "input": "raw_score without calibration",
            "purpose": "direct transfer baseline",
        },
        "M1_raw_only_platt": {
            "version": "M1",
            "input": "raw_score",
            "purpose": "ordinary calibration baseline",
        },
        "M2_behaviour_only_logistic": {
            "version": "M2",
            "input": "behaviour covariates only",
            "purpose": "ETH behaviour explanatory baseline",
        },
        "M3_raw_plus_behaviour_platt": {
            "version": "M3",
            "input": "raw_score + behaviour covariates",
            "purpose": "proposed behaviour-aware calibration",
        },
        "M4_weighted_raw_plus_behaviour": {
            "version": "M4",
            "input": "raw_score + behaviour + sample_weight",
            "purpose": "robustness check",
        },
    }

    all_repeat_rows = []
    summary_rows = []

    # Add V0 to summary for each n_pos for table readability
    for n_pos in N_POS_SIZES:
        v0_row = {
            "model": "V0_frozen_raw",
            "n_pos": n_pos,
            "auc_mean": v0_eval["auc"],
            "auc_std": 0.0,
            "auc_reversed_mean": v0_eval["auc_reversed"],
            "auc_reversed_std": 0.0,
            "f1_at_0.5_mean": v0_eval["f1_at_0.5"],
            "f1_at_0.5_std": 0.0,
            "f1_best_mean": v0_eval["f1_best"],
            "f1_best_std": 0.0,
            "brier_mean": v0_eval["brier"],
            "brier_std": 0.0,
            "ece_mean": v0_eval["ece"],
            "ece_std": 0.0,
            "n_repeats": 1,
        }
        v0_row.update(model_meta["V0_frozen_raw"])
        summary_rows.append(v0_row)

    models = [
        "M1_raw_only_platt",
        "M2_behaviour_only_logistic",
        "M3_raw_plus_behaviour_platt",
        "M4_weighted_raw_plus_behaviour",
    ]

    for n_pos in N_POS_SIZES:
        if n_pos >= int(y_true.sum()):
            continue

        for model_name in models:
            rows = run_one_model(
                model_name,
                raw_probs,
                behaviour_df,
                y_true,
                n_pos,
            )

            all_repeat_rows.extend(rows)

            agg = aggregate_rows(rows)
            agg["model"] = model_name
            agg["n_pos"] = n_pos
            agg.update(model_meta[model_name])

            summary_rows.append(agg)

            print(
                f"[{model_meta[model_name]['version']}] "
                f"n_pos={n_pos:2d} "
                f"AUC={agg.get('auc_mean', np.nan):.3f}±{agg.get('auc_std', np.nan):.3f} "
                f"F1@0.5={agg.get('f1_at_0.5_mean', np.nan):.3f} "
                f"F1best={agg.get('f1_best_mean', np.nan):.3f}±{agg.get('f1_best_std', np.nan):.3f} "
                f"Brier={agg.get('brier_mean', np.nan):.4f} "
                f"ECE={agg.get('ece_mean', np.nan):.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False)

    print_comparison_table(summary_df)

    result = {
        "config": {
            "negative_file": str(neg_file),
            "n_repeats": N_REPEATS,
            "n_pos_sizes": N_POS_SIZES,
            "max_negatives": MAX_NEGATIVES,
            "random_state": RANDOM_STATE,
        },
        "model_meta": model_meta,
        "v0_eval": v0_eval,
        "summary": summary_rows,
        "repeat_level_results": all_repeat_rows,
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n[Saved]")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_probs}")
    print("Done.")


if __name__ == "__main__":
    main()