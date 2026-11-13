
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from inspection_l_recovery import (
    load_elliptic,
    build_temporal_graphs,
    train_dgi,
    extract_embeddings,
    build_rf_input,
    train_rf,
    evaluate,
    TRAIN_RATIO,
    TEST_RATIO,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="/Volumes/T9/document/CYBR7901/gcn"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--trees", type=int, default=20)

    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Independent experiment directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("models") / f"run_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    print("\n=== BITCOIN TRAINING ===")

    # Load dataset
    df, edges = load_elliptic(args.data_dir)
    graphs, scaler = build_temporal_graphs(df, edges)

    train_graphs = graphs[:TRAIN_RATIO]
    test_graphs = graphs[
        TRAIN_RATIO:TRAIN_RATIO + TEST_RATIO
    ]

    # Train shared encoder
    encoder = train_dgi(
        train_graphs,
        input_dim=166,
        hidden_dim=128,
        epochs=args.epochs,
        lr=1e-4
    )

    # Save encoder and scaler
    torch.save(
        encoder.state_dict(),
        output / "bitcoin_encoder.pt"
    )

    joblib.dump(
        scaler,
        output / "bitcoin_scaler.pkl"
    )
    # Save DGI training loss history
    if hasattr(encoder, "loss_history"):
        loss_path = output / "dgi_loss_history.json"

        loss_path.write_text(
            json.dumps(encoder.loss_history, indent=2),
            encoding="utf-8"
        )

        print(f"[DGI] Loss history saved to: {loss_path}")
    else:
        raise RuntimeError(
            "DGI loss_history not found. "
            "Please update train_dgi() before training."
        )

    

    # Extract embeddings
    train_emb, train_y = extract_embeddings(
        encoder, train_graphs
    )
    test_emb, test_y = extract_embeddings(
        encoder, test_graphs
    )

    train_raw = np.vstack([
        g.x.numpy() for g in train_graphs
    ])

    test_raw = np.vstack([
        g.x.numpy() for g in test_graphs
    ])

    expected_dims = {
        "DNE": 128,
        "LF+DNE": 222,
        "AF+DNE": 294
    }

    results = {}

    # Train three RF configurations
    for mode, expected_dim in expected_dims.items():

        print(f"\n=== Random Forest: {mode} ===")

        X_train, y_train = build_rf_input(
            train_graphs,
            train_emb,
            train_y,
            train_raw,
            mode=mode
        )

        X_test, y_test = build_rf_input(
            test_graphs,
            test_emb,
            test_y,
            test_raw,
            mode=mode
        )

        assert X_train.shape[1] == expected_dim
        assert X_test.shape[1] == expected_dim

        rf = train_rf(
            X_train,
            y_train,
            n_estimators=args.trees,
            random_state=42
        )

        metrics = evaluate(
            rf,
            X_test,
            y_test,
            mode_name=mode
        )

        filename = mode.replace("+", "_")

        joblib.dump(
            rf,
            output / f"bitcoin_rf_{filename}.pkl"
        )

        results[mode] = {
            **{k: float(v) for k, v in metrics.items()},
            "input_features": expected_dim
        }

    config = {
        "dataset": "Elliptic Bitcoin",
        "epochs": args.epochs,
        "rf_trees": args.trees,
        "input_features": 166,
        "embedding_dimension": 128,
        "training_timesteps": TRAIN_RATIO,
        "testing_timesteps": TEST_RATIO,
        "scaler": "Training timesteps only",
        "random_seed": 42,
        "results": results
    }

    (output / "experiment.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    print("\nTRAINING COMPLETED")
    print("Models saved to:", output)


if __name__ == "__main__":
    main()
