import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.dataset import TreeHealthDataset


class TabularModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        cat_embedding_dims,
        num_numeric_features,
        hidden_dims=[128, 64],
        num_classes: int = 3,
        dropout_p: float = 0.5,
        negative_slope: float = 0.01,
    ):
        """
        Parameters:
            cat_cardinalities: List of unique counts for each categorical feature.
            cat_embedding_dims: List of embedding dimensions for each categorical feature.
            num_numeric_features: Number of continuous (numeric) features.
            hidden_dims: List of hidden layer sizes for the MLP.
            num_classes: Number of output classes.
            dropout_p: Dropout probability used for regularization.
            negative_slope: Negative slope for LeakyReLU to avoid "dead" neurons.
        """
        super(TabularModel, self).__init__()
        # Ensure the number of categorical features matches the number of embedding dimensions.
        assert len(cat_cardinalities) == len(
            cat_embedding_dims
        ), "The number of cardinalities must match the number of embedding dimensions."

        # Create embedding layers for categorical features.
        # Each embedding layer converts categorical indices into a dense vector representation.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
                for card, emb_dim in zip(cat_cardinalities, cat_embedding_dims)
            ]
        )
        # Calculate total dimension from all embeddings.
        total_emb_dim = sum(cat_embedding_dims)

        # The overall input dimension for the MLP is the sum of the embedding dimensions and the number of numeric features.
        self.input_dim = total_emb_dim + num_numeric_features

        # Build the MLP for joint processing of features.
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            # Linear layer to transform the input to the hidden dimension.
            layers.append(nn.Linear(in_dim, h_dim))
            # Batch Normalization to stabilize and accelerate training.
            layers.append(nn.BatchNorm1d(h_dim))
            # LeakyReLU introduces non-linearity and avoids "dying" neurons.
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            # Dropout for regularization, reducing overfitting by randomly deactivating neurons.
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        # The final layer outputs logits for each class.
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        """
        Forward pass.
            x_cat: Tensor of shape [batch_size, num_cat_features] containing categorical feature indices.
            x_num: Tensor of shape [batch_size, num_numeric_features] containing numeric features.
        """
        # Convert categorical indices to embeddings.
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Concatenate all embedding vectors to form a single feature vector
        # shape: [batch_size, total_emb_dim]
        x_cat_emb = torch.cat(embedded, dim=1)
        # Concatenate the embeddings with numeric features.
        # shape: [batch_size, input_dim]
        x = torch.cat([x_cat_emb, x_num], dim=1)
        # Pass the combined features through the MLP.
        return self.mlp(x)

    def train_epoch(
        self,
        train_loader,
        train_dataset: TreeHealthDataset,
        optimizer,
        criterion,
        device,
    ):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader providing batches of training data.
            train_dataset (TreeHealthDataset): The full training dataset, used to compute the average loss.
            optimizer (Optimizer): The optimizer used for updating model parameters.
            criterion (Loss): The loss function.
            device (torch.device): The device on which to perform training (e.g., "cpu" or "cuda").

        Returns:
            float: The average training loss for the epoch.
        """
        self.train()
        running_loss = 0.0
        for batch in train_loader:
            x_cat = batch["cat"].to(device)
            x_num = batch["num"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = self(x_cat, x_num)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_cat.size(0)
        epoch_loss = running_loss / len(train_dataset)
        return epoch_loss

    def validate_model(self, val_loader, device):
        """
        Evaluates the model on a validation dataset.

        Args:
            val_loader (DataLoader): DataLoader providing batches of validation data.
            device (torch.device): The device on which to perform validation.

        Returns:
            float: The validation accuracy.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x_cat = batch["cat"].to(device)
                x_num = batch["num"].to(device)
                targets = batch["target"].to(device)
                outputs = self(x_cat, x_num)
                _, preds = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        val_accuracy = correct / total
        return val_accuracy

    def predict(self, dataloader, device):
        """
        Generates predictions for the given dataset.

        Args:
            dataloader (DataLoader): DataLoader providing batches of data for inference.
            device (torch.device): The device on which to perform inference.

        Returns:
            np.ndarray: An array of predicted class indices.
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                x_cat = batch["cat"].to(device)
                x_num = batch["num"].to(device)
                outputs = self(x_cat, x_num)
                batch_predictions = torch.argmax(outputs, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
        return np.array(predictions)
