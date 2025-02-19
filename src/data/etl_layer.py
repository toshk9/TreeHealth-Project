import pandas as pd
import numpy as np

import pickle
import os
import json

from typing import List, Tuple, Dict, Any

from .dqc_layer import DQC

from src.api.schemas import InputBatch

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import prince


class ETL:
    """
    A class used to perform ETL (Extract, Transform, Load) operations on tree-related datasets.

    This class provides methods to clean, preprocess, and transform raw input data, vectorize categorical
    features, scale numerical features, and prepare or apply preprocessor objects. These operations are designed
    to ensure that the data is in a suitable format for further analysis or machine learning model predictions.
    """

    def check_problems_cols(
        self, row: pd.Series, problem_columns: List[str]
    ) -> pd.Series:
        """
        Processes a DataFrame row to aggregate problem indicators into a single 'problems' column.

        For each column in the specified problem columns (starting from the second column), the method checks
        if the value is 'Yes' and, if so, appends a formatted version of the column name to a cumulative string.
        If any column value is NaN, the 'problems' field is set to NaN and the row is returned immediately.
        Otherwise, the aggregated string (or "None" if no problems were found) is assigned to the 'problems' field.

        Args:
            row (pd.Series): A row from a DataFrame containing problem indicator columns.
            problem_columns (List[str]): A list of column names representing problem indicators. The first column
                is skipped in processing.

        Returns:
            pd.Series: The modified row with an added or updated "problems" field summarizing the detected issues.

        Example:
            >>> import pandas as pd
            >>> etl = ETL()
            >>> row = pd.Series({'col1': 'N/A', 'problem_one': 'Yes', 'problem_two': 'No'})
            >>> problem_cols = ['col1', 'problem_one', 'problem_two']
            >>> updated_row = etl.check_problems_cols(row, problem_cols)
            >>> print(updated_row["problems"])
            ProblemOne
        """
        problems = ""
        for column in problem_columns[1:]:
            problem = "".join([chunk.capitalize() for chunk in column.split("_")])
            if row[column] == "Yes":
                problems += problem + ","
            elif pd.isna(row[column]):
                row["problems"] = np.nan
                return row

        if not problems:
            problems = "None"
        else:
            problems = problems[:-1]

        row["problems"] = problems
        return row

    def data_processing(self, data: pd.DataFrame):
        """
        Processes raw tree data by cleaning, feature engineering, and filtering out outliers or inconsistent entries.

        The method performs the following steps:
            - Creates a copy of the input DataFrame.
            - Drops irrelevant columns.
            - Extracts 'month' and 'day' features from the 'created_at' timestamp and then drops 'created_at'.
            - Applies problem column aggregation using the check_problems_cols method.
            - Removes rows with 'tree_dbh' values outside of a realistic range (0 to 457).
            - Scales numerical features (currently 'tree_dbh') using the scale_num_features method.
            - Identifies and drops rows that do not pass the consistency and uniqueness check based on the 'health' column.

        Args:
            data (pd.DataFrame): The raw input DataFrame containing tree-related data.

        Returns:
            None

        Example:
            >>> import pandas as pd
            >>> etl = ETL()
            >>> raw_data = pd.read_csv("trees.csv")
            >>> etl.data_processing(raw_data)
            # The raw_data is processed internally (rows dropped, features engineered, etc.).
        """
        data = data.copy()
        data.drop(
            [
                "tree_id",
                "block_id",
                "census tract",
                "bin",
                "bbl",
                "borough",
                "spc_common",
                "zip_city",
                "nta_name",
                "council district",
                "state",
                "status",
                "community board",
                "stump_diam",
                "st_assem",
                "st_senate",
                "borocode",
                "cncldist",
                "y_sp",
                "x_sp",
                "latitude",
                "longitude",
            ],
            axis=1,
            inplace=True,
        )

        data["month"] = data["created_at"].dt.month
        data["day"] = data["created_at"].dt.day

        data.drop(["created_at"], axis=1, inplace=True)

        problem_cols = data.loc[:, "problems":"brch_other"].columns
        data.loc[:, problem_cols] = data.loc[:, problem_cols].apply(
            self.check_problems_cols, axis=1, problem_columns=problem_cols
        )

        min_tree_value = 0
        max_tree_value = 457  # dbh of the world's thickest tree
        outliers_idxs = data.loc[
            (data["tree_dbh"] < min_tree_value) | (data["tree_dbh"] > max_tree_value)
        ].index
        data.drop(index=outliers_idxs, inplace=True)

        data = self.scale_num_features(data, ["tree_dbh"])

        data_dqc = DQC()
        incost_idxs = data_dqc.consistency_uniqueness_check(
            data, consistency_columns=["health"]
        )
        data.drop(index=incost_idxs, inplace=True)

    def vectorize_features(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms categorical features in the processed DataFrame into numerical representations.

        The method performs the following operations:
            - Applies Multiple Correspondence Analysis (MCA) on a group of problem indicator columns,
              replacing them with a single 'problems' feature and saving the MCA model.
            - Applies MCA on target group conditions (e.g., 'steward', 'guards', 'sidewalk', 'curb_loc'),
              replacing them with a single 'tg_conditions' feature and saving the corresponding MCA model.
            - Performs label encoding on object-type columns and selected numeric categorical features,
              saving each label encoder to disk.

        Args:
            processed_data (pd.DataFrame): A DataFrame that has been preprocessed and is ready for feature vectorization.

        Returns:
            pd.DataFrame: A new DataFrame with vectorized and encoded features suitable for model training or prediction.

        Example:
            >>> import pandas as pd
            >>> etl = ETL()
            >>> processed_df = pd.read_csv("processed_trees.csv")
            >>> vectorized_df = etl.vectorize_features(processed_df)
            >>> vectorized_df.head()
        """
        data_vec = processed_data.copy()

        mca = prince.MCA(n_components=1, random_state=42)
        mca_problems = mca.fit_transform(data_vec.loc[:, "root_stone":"brch_other"])

        data_vec["problems"] = mca_problems
        data_vec.drop(
            [
                "root_stone",
                "root_grate",
                "root_other",
                "trunk_wire",
                "trnk_light",
                "trnk_other",
                "brch_light",
                "brch_shoe",
                "brch_other",
            ],
            axis=1,
            inplace=True,
        )

        with open("models/preprocessors/mca_problems.pkl", "wb") as f:
            pickle.dump(mca, f)

        mca = prince.MCA(n_components=1, random_state=42)
        mca_tg_conditions = mca.fit_transform(
            data_vec[["steward", "guards", "sidewalk", "curb_loc"]]
        )

        data_vec["tg_conditions"] = mca_tg_conditions
        data_vec.drop(
            ["steward", "guards", "sidewalk", "curb_loc"], axis=1, inplace=True
        )

        with open("models/preprocessors/mca_tg_conditions.pkl", "wb") as f:
            pickle.dump(mca, f)

        numeric_cat_features = [
            "postcode",
            "borocode",
            "cncldist",
            "st_assem",
            "st_senate",
            "boro_ct",
            "month",
            "day",
        ]

        for column, dtype in processed_data.dtypes.items():
            if dtype == np.object_:
                le = LabelEncoder()
                data_vec[column] = le.fit_transform(processed_data[column])

                with open(
                    f"models/preprocessors/label_encoders/{column}_label_encoder.pkl",
                    "wb",
                ) as f:
                    pickle.dump(le, f)

            elif column in numeric_cat_features:
                print(column)

                le = LabelEncoder()
                data_vec[column] = le.fit_transform(processed_data[column])

                with open(
                    f"models/preprocessors/label_encoders/{column}_label_encoder.pkl",
                    "wb",
                ) as f:
                    pickle.dump(le, f)

        return data_vec

    def scale_num_features(
        self, data: pd.DataFrame, num_columns: List[str] = ["tree_dbh"]
    ) -> pd.DataFrame:
        """
        Scales specified numerical features in the DataFrame using MinMax scaling.

        If a pre-fitted scaler exists on disk, it is loaded and applied to the specified numeric columns.
        Otherwise, a new MinMaxScaler is fitted on the data and saved for future use.

        Args:
            data (pd.DataFrame): The input DataFrame containing numerical features.
            num_columns (List[str], optional): A list of column names to be scaled. Defaults to ["tree_dbh"].

        Returns:
            pd.DataFrame: A DataFrame with the specified numerical columns scaled to the range [0, 1].

        Example:
            >>> import pandas as pd
            >>> etl = ETL()
            >>> df = pd.DataFrame({'tree_dbh': [10, 20, 30]})
            >>> scaled_df = etl.scale_num_features(df)
            >>> scaled_df.head()
        """
        data = data.copy()
        if os.path.exists("models/preprocessors/scaler.pkl"):
            scaler, _, _ = self.prepare_preprocessors()
            num_features_scaled = scaler.transform(data[num_columns])
        else:
            scaler = MinMaxScaler()
            num_features_scaled = scaler.fit_transform(data[num_columns])

            with open(f"models/preprocessors/scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        data[num_columns] = num_features_scaled
        return data

    def prepare_preprocessors(
        self, processors_path: str = "models\preprocessors"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Loads and returns preprocessor objects (scaler, label encoders, and MCA models) from disk.

        The method reads a scaler, a set of label encoders for specified columns, and MCA models for feature
        transformation from designated files located in the provided processors path.

        Args:
            processors_path (str, optional): The base directory path where preprocessor objects are stored.
                Defaults to "models/preprocessors".

        Returns:
            Tuple[Any, Dict[str, Any], Dict[str, Any]]:
                - scaler: The loaded scaler object.
                - label_encoders (Dict[str, Any]): A dictionary mapping column names to their respective label encoder objects.
                - mcas (Dict[str, Any]): A dictionary mapping feature groups (e.g., "problems", "tg_conditions") to their MCA objects.

        Example:
            >>> etl = ETL()
            >>> scaler, encoders, mcas = etl.prepare_preprocessors()
        """
        with open(os.path.join(processors_path, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        encoder_files = {
            "spc_latin": "spc_latin_label_encoder.pkl",
            "user_type": "user_type_label_encoder.pkl",
            "address": "address_label_encoder.pkl",
            "postcode": "postcode_label_encoder.pkl",
            "nta": "nta_label_encoder.pkl",
            "boro_ct": "boro_ct_label_encoder.pkl",
            "month": "month_label_encoder.pkl",
            "day": "day_label_encoder.pkl",
            "health": "health_label_encoder.pkl",
        }

        label_encoders = {}
        for col, filename in encoder_files.items():
            path = os.path.join(processors_path, "label_encoders", filename)
            with open(path, "rb") as f:
                label_encoders[col] = pickle.load(f)

        mca_files = {
            "problems": "mca_problems.pkl",
            "tg_conditions": "mca_tg_conditions.pkl",
        }

        mcas = {}
        for col, filename in mca_files.items():
            path = os.path.join(processors_path, filename)
            with open(path, "rb") as f:
                mcas[col] = pickle.load(f)

        return scaler, label_encoders, mcas

    def data_preprocessing(self, input_batch: InputBatch) -> pd.DataFrame:
        """
        Preprocesses the input batch data and transforms it into a DataFrame suitable for model predictions.

        The method performs the following steps:
            - Converts the input batch to JSON and loads it into a Python dictionary.
            - Creates a DataFrame from the input batch and renames columns as needed.
            - Loads preprocessor objects (scaler, label encoders, and MCA models) using prepare_preprocessors.
            - Scales numerical features and transforms problem-related columns using MCA.
            - Extracts date features ('month' and 'day') from the 'mapping_date' column.
            - Applies MCA to transform target group condition features.
            - Applies label encoding to categorical features.

        Args:
            input_batch (InputBatch): An object containing the raw input data batch. The object should support
                model_dump_json() method to serialize the data.

        Returns:
            pd.DataFrame: A preprocessed DataFrame with transformed numerical and categorical features ready for prediction.

        Example:
            >>> etl = ETL()
            >>> preprocessed_df = etl.data_preprocessing(input_batch)
            >>> preprocessed_df.head()
        """
        input_batch = input_batch.model_dump_json()
        input_batch = json.loads(input_batch)
        input_batch = input_batch["input_batch"]

        input_df = pd.DataFrame(input_batch)

        input_df.rename(
            columns={
                "scientific_name": "spc_latin",
                "censustract": "boro_ct",
                "curb_location": "curb_loc",
            },
            inplace=True,
        )

        scaler, label_encoders, mcas = self.prepare_preprocessors()

        input_df["tree_dbh"] = scaler.transform(pd.DataFrame(input_df["tree_dbh"]))

        problems_cols = [
            "root_stone",
            "root_grate",
            "root_other",
            "trunk_wire",
            "trnk_light",
            "trnk_other",
            "brch_light",
            "brch_shoe",
            "brch_other",
        ]

        problems_processed = mcas["problems"].transform(input_df.loc[:, problems_cols])
        input_df.drop(problems_cols, axis=1, inplace=True)
        input_df.insert(
            input_df.columns.get_loc("user_type") + 1, "problems", problems_processed
        )

        input_df["mapping_date"] = pd.to_datetime(
            input_df["mapping_date"], format="%Y-%m-%d"
        )

        input_df.insert(
            input_df.columns.get_loc("boro_ct") + 1,
            "month",
            input_df["mapping_date"].dt.month,
        )
        input_df.insert(
            input_df.columns.get_loc("month") + 1,
            "day",
            input_df["mapping_date"].dt.day,
        )
        input_df.drop(["mapping_date"], axis=1, inplace=True)

        tg_conditions_cols = ["steward", "guards", "sidewalk", "curb_loc"]
        input_df["tg_conditions"] = mcas["tg_conditions"].transform(
            input_df.loc[:, tg_conditions_cols]
        )
        input_df.drop(tg_conditions_cols, axis=1, inplace=True)

        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        return input_df

    def data_postprocessing(
        self, model_predictions: np.array, input_batch: InputBatch
    ) -> Dict[str, Dict[str, Any]]:
        """
        Postprocesses the model predictions by mapping them back to original labels and merging with the input batch.

        The method performs the following steps:
            - Loads label encoders (specifically for the 'health' feature) using prepare_preprocessors.
            - Converts the input batch to a dictionary.
            - Applies the inverse transformation on model predictions to obtain the original 'health' labels.
            - Updates each entry in the input batch with the corresponding 'health' prediction.

        Args:
            model_predictions (np.array): An array of numerical model predictions for the 'health' feature.
            input_batch (InputBatch): The original input batch containing raw data. Must support model_dump_json().

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with a key "output_batch" that holds the updated input data,
                where the 'health' field has been replaced by the corresponding predicted label.

        Example:
            >>> etl = ETL()
            >>> predictions = np.array([0, 1, 2])
            >>> output = etl.data_postprocessing(predictions, input_batch)
            >>> print(output["output_batch"][0]["health"])
        """
        _, encoders, _ = self.prepare_preprocessors()

        input_batch = input_batch.model_dump_json()
        input_batch = json.loads(input_batch)

        output_batch = {}

        output_batch["output_batch"] = input_batch["input_batch"]

        predictions_processed = encoders["health"].inverse_transform(model_predictions)

        for i, _ in enumerate(output_batch["output_batch"]):
            output_batch["output_batch"][i]["health"] = predictions_processed.tolist()[
                i
            ]

        return output_batch
