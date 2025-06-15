import os
import copy
import pandas as pd
from spectral_based_prediction.constants_config import DATA_FOLDER_PATH, ColumnName


class Dataset:
    def __init__(self, train_file_name, validation_file_name, test_file_name, data_folder_spec, target_variables):
        """
        Initializes the Dataset class to work with scaled and non-scaled data.

        :param train_file_name: File name of the scaled training data (e.g., *_plsr.parquet).
        :param validation_file_name: File name of the scaled validation data.
        :param test_file_name: File name of the scaled test data.
        :param data_folder_spec: Folder where data is stored.
        :param target_variables: List of target (Y) variable names.
        """
        self.data_folder_spec = data_folder_spec
        self.target_variables = target_variables

        # Load scaled and non-scaled data for train, validation, and test splits
        self.ID_train, self.X_train, self.Y_train, self.Y_train_non_scaled = self.load_and_merge(
            scaled_file_name=train_file_name, non_scaled_file_name="train_data.parquet"
        )
        self.ID_val, self.X_val, self.Y_val, self.Y_val_non_scaled = self.load_and_merge(
            scaled_file_name=validation_file_name, non_scaled_file_name="validation_data.parquet"
        )
        self.ID_test, self.X_test, self.Y_test, self.Y_test_non_scaled = self.load_and_merge(
            scaled_file_name=test_file_name, non_scaled_file_name="test_data.parquet"
        )

    def load_and_merge(self, scaled_file_name, non_scaled_file_name):
        """
        Loads scaled data and merges it with the non-scaled Y values on the ID column.

        :param scaled_file_name: Scaled data file name (e.g., *_plsr.parquet).
        :param non_scaled_file_name: Non-scaled data file name (e.g., test_data.parquet).
        :return: Tuple (ID, X, Y_scaled, Y_non_scaled).
        """
        # Define paths for scaled and non-scaled files
        data_folder = os.path.join(DATA_FOLDER_PATH, self.data_folder_spec)
        scaled_file_path = os.path.join(data_folder, scaled_file_name)
        non_scaled_file_path = os.path.join(data_folder, non_scaled_file_name)

        # Load the scaled data
        scaled_data = pd.read_parquet(scaled_file_path).reset_index(drop=True)
        #
        # # Extract the ID column
        # id_col = scaled_data[ColumnName.id.value].reset_index(drop=True)
        #
        # # Extract scaled features (X) and targets (Y)
        # X = scaled_data.drop(columns=self.target_variables).reset_index(drop=True)
        # Y_scaled = scaled_data[self.target_variables].reset_index(drop=True)
        #
        # # If no scaler is provided, return only the scaled data (no merging)
        # if self.y_scaler is None:
        #     return id_col, X.drop(columns=ColumnName.id.value), Y_scaled, None

        # Load the non-scaled data
        non_scaled_data = pd.read_parquet(non_scaled_file_path).reset_index(drop=True)

        # Merge scaled and non-scaled data on the ID column
        merged_data = scaled_data.merge(
            non_scaled_data[[ColumnName.id.value] + self.target_variables],
            on=ColumnName.id.value,
            how="left"
        )

        id_col = merged_data[[ColumnName.id.value]]

        if len(self.target_variables) == 1:
            Y_scaled = merged_data[[f'{self.target_variables[0]}_x']].rename(
                columns={f'{self.target_variables[0]}_x': self.target_variables[0]}
            ).reset_index(drop=True)
            Y_non_scaled = merged_data[[f'{self.target_variables[0]}_y']].rename(
                columns={f'{self.target_variables[0]}_y': f'{self.target_variables[0]}_non_scaled'}
            ).reset_index(drop=True)

            # Extract relevant feature columns by removing ID and target columns
            excluded_columns = {ColumnName.id.value, f'{self.target_variables[0]}_x', f'{self.target_variables[0]}_y'}
        else:
            Y_scaled = merged_data[
                [f'{var}_x' for var in self.target_variables]
            ].rename(columns={f'{var}_x': var for var in self.target_variables}).reset_index(drop=True)

            Y_non_scaled = merged_data[
                [f'{var}_y' for var in self.target_variables]
            ].rename(columns={f'{var}_y': f'{var}_non_scaled' for var in self.target_variables}).reset_index(drop=True)

            # Extract relevant feature columns by removing ID and all target-related columns
            excluded_columns = {ColumnName.id.value}.union(
                {f'{var}_x' for var in self.target_variables}, {f'{var}_y' for var in self.target_variables}
            )

        relevant_feature_columns = [col for col in merged_data.columns if col not in excluded_columns]
        X = merged_data[relevant_feature_columns].reset_index(drop=True)

        # Return the separated data
        return (
            id_col.reset_index(drop=True),
            X.reset_index(drop=True),
            Y_scaled.reset_index(drop=True),
            Y_non_scaled.reset_index(drop=True)
        )

    def copy(self, deep=False):
        """
        Creates a copy of the Dataset instance.

        :param deep: If True, creates a deep copy. Otherwise, creates a shallow copy.
        :return: A copy of the Dataset object.
        """
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)
