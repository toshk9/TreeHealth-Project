#!/usr/bin/env python

import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import TreeHealthDataset
from src.model import TabularModel
from src.utils.hyperoptimization import load_best_trial

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from typing import List, Optional


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data file or directory.",
    )
    parser.add_argument(
        "--val_data",
        type=Optional[str],
        default=None,
        required=False,
        help="Path to the validation data file or directory.",
    )
    parser.add_argument(
        "--cat_features",
        type=List[str],
        default=[
            "spc_latin",
            "user_type",
            "address",
            "postcode",
            "nta",
            "boro_ct",
            "month",
            "day",
        ],
        required=False,
        help="List of categorical feature names to be used from the dataset.",
    )
    parser.add_argument(
        "--num_features",
        type=Optional[List[str]],
        default=["tree_dbh", "problems", "tg_conditions"],
        required=False,
        help="List of numerical feature names to be used from the dataset.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="health",
        required=False,
        help="Name of the target column in the dataset.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Size of the test dataset.", required=False
    )
    parser.add_argument(
        "--model_file", type=str, required=False, default=None, help="Path to saved model."
    )
    parser.add_argument(
        "--model_save",
        type=str,
        default="models/torch_models/model_v1.pth",
        help="Path to save model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train the model.", required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for DataLoader.", required=False
    )
    parser.add_argument(
        "--hyperparam_file",
        type=str,
        required=False,
        default=None,
        help="File with hyperparameters for the model in json format.",
        required=False,
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer.", required=False
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization).",
        required=False
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cpu' or 'cuda').",
        required=False
    )

    args = parser.parse_args()

    train_data = pd.read_csv(args.train_data)

    X = train_data[args.cat_features + args.num_features]
    y = train_data[args.target_column]

    if args.val_data:
        val_data = pd.read_csv(args.val_data)
        X_val = val_data[args.cat_features + args.num_features]
        y_val = val_data[args.target_column]

        X_train, y_train = X, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )

    if args.hyperparam_file:
        best_hyperparams = load_best_trial(args.hyperparam_file)[
            "result"]["params"]
        CAT_EMBEDDING_DIMS = [
            int(best_hyperparams[f"cat_emb_dim_{i}"]) for i in range(8)
        ]
        HIDDEN_DIMS = [int(best_hyperparams["hidden_dim"])]
        DROPOUT_P = best_hyperparams["dropout_p"]
        BATCH_SIZE = int(best_hyperparams["batch_size"])
        LEARNING_RATE = best_hyperparams["learning_rate"]
        WEIGHT_DECAY = best_hyperparams["weight_decay"]
    else:
        BATCH_SIZE = args.batch_size
        LEARNING_RATE = args.lr
        WEIGHT_DECAY = args.weight_decay

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    train_dataset = TreeHealthDataset(
        train_df, args.cat_features, args.num_features, args.target_column
    )
    val_dataset = TreeHealthDataset(
        val_df, args.cat_features, args.num_features, args.target_column
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if args.model_file:
        model = torch.load(args.model_file, weights_only=False)
    else:
        cat_cardinalities = [
            train_data.nunique()[feature] for feature in args.cat_features
        ]
        num_numeric_features = len(args.num_features)
        num_classes = train_data[args.target_column].nunique()

        model = (
            TabularModel(
                cat_cardinalities=cat_cardinalities,
                cat_embedding_dims=CAT_EMBEDDING_DIMS,
                num_numeric_features=num_numeric_features,
                hidden_dims=HIDDEN_DIMS,
                num_classes=num_classes,
                dropout_p=DROPOUT_P,
            )
            if args.hyperparam_file
            else TabularModel(
                cat_cardinalities=cat_cardinalities,
                num_numeric_features=num_numeric_features,
                num_classes=num_classes,
            )
        )
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    y_train_np = y_train.values
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_np), y=y_train_np
    )
    class_weights = torch.tensor(
        class_weights, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.train_model(
        args.epochs,
        train_loader,
        train_dataset,
        optimizer,
        criterion,
        val_loader,
        device,
        verbose=True,
    )

    torch.save(model, args.model_save)


if __name__ == "__main__":
    main()
