import pandas as pd
import numpy as np

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


class EDA:
    """
    Exploratory Data Analysis (EDA) class for performing various statistical checks and visualizations on a pandas DataFrame.

    Methods:
    - features_corr_check: Checks for correlations between specified features using various methods and visualizes them using a heatmap.
    - cramers_v: calculating cramers v coefficient for features correlation measurement
    """

    def features_corr_check(
        self,
        df: pd.DataFrame,
        columns: List[str],
        methods: list = ["pearson"],
        **kwargs,
    ) -> None:
        """
        Checks for and visualizes the correlation between specified columns in the DataFrame using specified methods.

        This method calculates the correlation between the specified columns using one or more correlation methods
        (e.g., Pearson, Kendall, Spearman) and visualizes the correlation matrix using a heatmap. This can be useful
        for identifying potential relationships between variables that may warrant further investigation.

        Parameters:
        - columns (List[str]): A list of column names within the DataFrame for which correlations are to be calculated.
        - methods (list, optional): A list of strings indicating the correlation methods to use. Default is ["pearson"].
                                Other valid options include "kendall" and "spearman".
        - **kwargs: Additional keyword arguments to be passed to the pandas `corr` method.

        Returns:
        - None: This method does not return a value but displays a heatmap visualization of the correlation matrix for
                each specified method.

        Example:
        ```python
        eda = EDA()
        eda.features_corr_check(my_dataframe, columns=['feature1', 'feature2', 'feature3'], methods=['pearson', 'spearman'])
        ```
        This will calculate and display the correlation matrix for 'feature1', 'feature2', and 'feature3' using both the
        Pearson and Spearman correlation coefficients.
        """
        for method in methods:
            method = method.lower()
            df_corr: pd.DataFrame = df[columns].corr(method=method, **kwargs)
            plt.figure(figsize=(25, 25))
            sns.heatmap(df_corr, annot=True, fmt=".3f")
            plt.title(f"Checking for correlation using {method} method", fontsize=20)
            plt.show()

    def cramers_v(confusion_matrix: pd.DataFrame) -> np.array:
        """
        Calculates Cramér's V statistic to measure the association between two categorical variables
        based on a contingency table.

        This function computes the chi-squared statistic from the provided confusion matrix (i.e.,
        contingency table) using the chi2_contingency test from scipy.stats. It then calculates the
        Cramér's V value, which is a normalized measure of association ranging from 0 (no association)
        to 1 (perfect association). The calculation includes corrections for bias due to sample size
        and the shape of the table.

        Parameters:
        - confusion_matrix (pd.DataFrame): A DataFrame representing the contingency table where each
        cell indicates the frequency of occurrences for the combination of categories for the two
        variables under analysis.

        Returns:
        - np.array: A numerical value representing the Cramér's V statistic computed from the confusion matrix.

        Example:
        ```python
        import pandas as pd
        import numpy as np
        from scipy import stats

        # Example contingency table
        confusion_matrix = pd.DataFrame([[30, 10, 20],
                                        [20, 15, 25],
                                        [10, 20, 30]])

        cv = cramers_v(confusion_matrix)
        print("Cramér's V:", cv)
        ```
        """
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rcorr = r - (r - 1) ** 2 / (n - 1)
        kcorr = k - (k - 1) ** 2 / (n - 1)
        return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))
