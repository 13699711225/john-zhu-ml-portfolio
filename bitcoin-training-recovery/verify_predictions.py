from pathlib import Path
import json

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from inspection_l_recovery import (
    load_elliptic,
    build_temporal_graphs,
    extract_embeddings,
    build_rf_input,
    GINEncoder,
    TRAIN_RATIO,
    TEST_RATIO,
)

DATA_DIR = "/Volumes/T9/document/CYBR7901/gcn"
RUN_DIR = Path("models/run_20261113_225421")

# 1. Read saved experiment
config = json.loads(
    (RUN_DIR / "experiment.json").read_text(encoding="utf-8")
)

# 2. Reload trained encoder
encoder = GINEncoder(input_dim=166, hidden_dim=128)

state = torch.load(
    RUN_DIR / "bitcoin_encoder.pt",
    map_location="cpu",
    weights_only=True,
)
encoder.load_state_dict(state, strict=True)
encoder.eval()

# 3. Reload scaler
saved_scaler = joblib.load(
    RUN_DIR / "bitcoin_scaler.pkl"
)

# 4. Rebuild exactly the same temporal graphs
df, edges = load_elliptic(DATA_DIR)
graphs, rebuilt_scaler = build_temporal_graphs(df, edges)

np.testing.assert_allclose(
    rebuilt_scaler.mean_,
    saved_scaler.mean_,
    rtol=1e-10,
    atol=1e-10,
)
np.testing.assert_allclose(
    rebuilt_scaler.scale_,
    saved_scaler.scale_,
    rtol=1e-10,
    atol=1e-10,
)

test_graphs = graphs[
    TRAIN_RATIO:TRAIN_RATIO + TEST_RATIO
]

# 5. Generate embeddings with reloaded encoder
embeddings, labels = extract_embeddings(
    encoder, test_graphs
)

raw_features = np.vstack([
    graph.x.cpu().numpy()
    for graph in test_graphs
])

print("\n=== PREDICTION REPRODUCIBILITY TEST ===")

expected_dimensions = {
    "DNE": 128,
    "LF+DNE": 222,
    "AF+DNE": 294,
}

# 6. Reload three RF models and reproduce metrics
for mode, expected_dim in expected_dimensions.items():
    filename = mode.replace("+", "_")

    rf = joblib.load(
        RUN_DIR / f"bitcoin_rf_{filename}.pkl"
    )

    X_test, y_test = build_rf_input(
        test_graphs,
        embeddings,
        labels,
        raw_features,
        mode=mode,
    )

    assert X_test.shape[1] == expected_dim

    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test)[:, 1]

    actual = {
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "auc": roc_auc_score(
            y_test, probabilities
        ),
    }

    expected = config["results"][mode]

    print(f"\n{mode}")

    for metric, value in actual.items():
        original = float(expected[metric])

        print(
            f"{metric}: "
            f"reloaded={value:.6f}, "
            f"original={original:.6f}"
        )

        assert np.isclose(
            value,
            original,
            rtol=1e-6,
            atol=1e-6,
        ), f"{mode}: {metric} mismatch"

    print(f"[OK] {mode} reproduced successfully")

print("\nPREDICTION REPRODUCIBILITY PASSED")
