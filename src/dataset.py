import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from typing import List, Dict, Optional


class TreeHealthDataset(Dataset):
    """
    A custom Dataset for tree health data.

    This dataset organizes data from a pandas DataFrame into separate tensors for categorical and numerical
    features, and optionally includes a target variable if provided. The categorical features are expected to be
    pre-encoded as integers (e.g., using LabelEncoder), while numerical features are converted to float tensors.
    This class is designed to work seamlessly with PyTorch's DataLoader for batch processing during training or inference.

    Attributes:
        target_column (str or None): The name of the target variable column. If None, the dataset is considered unlabeled.
        X_cat (np.ndarray): A NumPy array containing the categorical features, cast to int64.
        X_num (np.ndarray): A NumPy array containing the numerical features, cast to float32.
        y (np.ndarray or None): A NumPy array containing the target values, cast to int64 if target_column is provided;
                                otherwise, None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_features: List[str] = [
            "spc_latin",
            "user_type",
            "address",
            "postcode",
            "nta",
            "boro_ct",
            "month",
            "day",
        ],
        num_features: List[str] = ["tree_dbh", "problems", "tg_conditions"],
        target_column: str = None,
    ):
        """
        Initializes the TreeHealthDataset instance.

        This constructor extracts categorical and numerical features from the provided DataFrame and converts them
        to NumPy arrays with appropriate data types. If a target column is specified, it also extracts and converts
        the target values.

        Args:
            df (pd.DataFrame): A DataFrame containing the features and optionally the target variable.
            cat_features (List[str], optional): A list of column names representing categorical features (pre-encoded
                as integers). Defaults to ["spc_latin", "user_type", "address", "postcode", "nta", "boro_ct", "month", "day"].
            num_features (List[str], optional): A list of column names representing numerical features.
                Defaults to ["tree_dbh", "problems", "tg_conditions"].
            target_column (str, optional): The name of the target variable column. If not provided, the dataset
                will be considered unlabeled. Defaults to None.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "spc_latin": [0, 1],
            ...     "user_type": [1, 0],
            ...     "address": [2, 3],
            ...     "postcode": [4, 5],
            ...     "nta": [6, 7],
            ...     "boro_ct": [8, 9],
            ...     "month": [10, 11],
            ...     "day": [12, 13],
            ...     "tree_dbh": [14.0, 15.0],
            ...     "problems": [16.0, 17.0],
            ...     "tg_conditions": [18.0, 19.0],
            ...     "target": [0, 1]
            ... })
            >>> dataset = TreeHealthDataset(df, target_column="target")
        """
        self.target_column = target_column
        self.X_cat = df[cat_features].values.astype(np.int64)
        self.X_num = df[num_features].values.astype(np.float32)
        if target_column:
            self.y = df[target_column].values.astype(np.int64)
        else:
            self.y = None

    def __len__(self) -> Optional[int]:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples (rows) in the dataset.

        Example:
            >>> len(dataset)
            100
        """
        return len(self.X_cat)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the sample at the given index.

        This method returns a dictionary containing the categorical and numerical feature tensors for the sample.
        If a target column was provided during initialization, the dictionary will also include the target tensor.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - 'cat': A tensor of categorical features (dtype: torch.long).
                - 'num': A tensor of numerical features (dtype: torch.float).
                - 'target' (optional): A tensor of the target label (dtype: torch.long) if target_column is defined.

        Example:
            >>> sample = dataset[0]
            >>> sample['cat'].shape
            torch.Size([8])
            >>> sample['num'].shape
            torch.Size([3])
            >>> sample['target']
            tensor(0)
        """
        if self.target_column:
            return {
                "cat": torch.tensor(self.X_cat[idx], dtype=torch.long),
                "num": torch.tensor(self.X_num[idx], dtype=torch.float),
                "target": torch.tensor(self.y[idx], dtype=torch.long),
            }
        else:
            return {
                "cat": torch.tensor(self.X_cat[idx], dtype=torch.long),
                "num": torch.tensor(self.X_num[idx], dtype=torch.float),
            }
