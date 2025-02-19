import pandas as pd

from typing import List


class DQC:
    """
    Data Quality Control (DQC) class for performing various data cleaning and quality assurance operations
    on a given DataFrame.

    :param df: The DataFrame to perform quality control operations on.
    :param df_title: Optional title for the DataFrame.

    The DQC class allows users to conduct quality control checks on a DataFrame,
    such as identifying missing values, assessing data consistency. Once initialized, users can utilize various methods provided
    by the class to analyze and improve the quality of the data.
    """

    def data_review(self, df: pd.DataFrame, columns2descr: List[str]) -> None:
        """
        Prints a review of the dataset.

        This method prints various information about the dataset, including:
        - Dataset title (if provided)
        - DataFrame info, which includes the data types, non-null counts, and memory usage
        - Descriptive statistics such as count, mean, standard deviation, min, and max values
        - A snippet of the dataset showing the first 5 rows

        It serves as a quick overview of the dataset's structure and contents.
        """
        print(df.info())
        print("\n", "-" * 50)

        print("\nDescriptive data statistics:\n")
        print(df.loc[:, columns2descr].describe())
        print("\n", "-" * 50)

        print("\nChunk of the dataset:\n")
        print(df.head(5))
        print("\n", "-" * 50)

    def na_values_check(self, df: pd.DataFrame) -> pd.Index:
        """
        Checks for rows with missing values in the DataFrame.

        :return: A pandas Int64Index containing the indexes of rows with missing values.

        This method identifies rows in the DataFrame that contain one or more missing values (NaNs).
        It returns the indexes of these rows for further investigation or handling.
        """
        na_rows: pd.core.frame.DataFrame = df[df.isna().any(axis=1)]
        na_rows_idx: pd.Index = na_rows.index

        print(f"\nNumber of rows with missing values: {len(na_rows_idx)}")
        print(f"\nDetails of rows with missing values:")
        print(df.isnull().sum())
        return na_rows_idx

    def consistency_uniqueness_check(
        self, df: pd.DataFrame, consistency_columns: List[str]
    ) -> pd.Index:
        """
        Checks for consistency and uniqueness of specified columns in the DataFrame.

        :param consistency_columns: A list of column names to check for consistency and uniqueness.
        :return: A pandas Int64Index containing the indexes of rows with conflicting or duplicated data.

        This method checks for conflicting or duplicated rows in the DataFrame based on specified columns.
        It returns the indexes of rows where inconsistencies or duplications are found.
        """

        for column in consistency_columns:
            if column not in df.columns:
                raise f"The specified column {column} does not exist in the Dataset"

        columns_to_consistency_check: List[str] = [
            column for column in df.columns if column not in consistency_columns
        ]

        ds_without_the_feature = df.loc[:, columns_to_consistency_check]
        inconsistent_rows = ds_without_the_feature[ds_without_the_feature.duplicated()]
        inconsistent_rows_idx = inconsistent_rows.index

        print(
            f"\nNumber of conflicting or duplicated rows: {len(inconsistent_rows_idx)}"
        )
        return inconsistent_rows_idx
