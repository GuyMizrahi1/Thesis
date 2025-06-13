import os
import random
import joblib
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold
# from sklearn.exceptions import NotFittedError
# from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from spectral_based_prediction.constants_config import TARGET_VARIABLES, NON_FEATURE_COLUMNS, MEAN


class XGBoostGeneric:
    def __init__(self, model_name, target_variables, n_splits=2, save_dir='models/', save_figure_dir='figures/',
                 is_multi_output=True, y_scaler=None):
        """
        Initialize XGBoost model

        Args:
            model_name: Name of the model
            target_variables: List of target variable names
            is_multi_output: Boolean indicating if the model predicts multiple targets
            save_dir: Directory to save model and results
            save_figure_dir: Directory to save figures
        """
        self.final_val_rmse = {}
        self.final_train_rmse = {}
        self.model_name = model_name
        self.target_variables = target_variables if isinstance(target_variables, list) else [target_variables]
        self.is_multi_output = is_multi_output
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.n_splits = n_splits
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_figure_dir = save_figure_dir
        os.makedirs(self.save_figure_dir, exist_ok=True)
        self.model = None
        self.best_params = None
        self.train_rmses = {}
        self.val_rmses = {}
        self.evaluated_val_rmses = {}
        self.evaluated_test_rmses = {}
        self.targets_rmses_for_best_params = {}
        self.scalers = {}
        # self.X_scaler = None
        self.y_scaler = y_scaler

    def preprocess_data(self, dataset="train"):
        """Preprocess data differently for regular XGBoost and PLSR-based XGBoost"""
        feature_columns = [col for col in self.train_data.columns if col not in NON_FEATURE_COLUMNS]

        # Get the right dataset
        if dataset == "train":
            X = self.train_data[feature_columns]
            y = self.train_data[self.target_variables]
        elif dataset == "val":
            X = self.val_data[feature_columns]
            y = self.val_data[self.target_variables]
        else:  # test
            X = self.test_data[feature_columns]
            y = self.test_data[self.target_variables]

        # # Only apply scaling for regular XGBoost (not PLSR)
        # if not self.model_name.endswith('_plsr'):
        # #     if dataset == "train":
        # #         # Fit and transform on training data
        # #         X = self.X_scaler.fit_transform(X)
        # #         if self.y_scaler is not None:
        # #             y_reshaped = y.values.reshape(-1, 1) if not self.is_multi_output else y.values
        # #             y = self.y_scaler.fit_transform(y_reshaped)
        # #     else:
        # #         # Only transform validation/test data
        # #         X = self.X_scaler.transform(X)
        # #         if self.y_scaler is not None:
        # #             y_reshaped = y.values.reshape(-1, 1) if not self.is_multi_output else y.values
        # #             y = self.y_scaler.transform(y_reshaped)
        #
        # # Fit scaler on training targets but do NOT scale them
        # if not self.model_name.endswith('_plsr') and dataset == "train":
        #     if self.y_scaler is not None:
        #         y_array = y.values.reshape(-1, 1) if not self.is_multi_output else y.values
        #         self.y_scaler.fit(y_array)  # Fit only, no transform
        #
        # # For single output, return y as a 1D array
        # if not self.is_multi_output:
        #     y = y[self.target_variables[0]] if isinstance(y, pd.DataFrame) else y.ravel()

        return X, y

    def extract_target_values(self, y, target):
        if isinstance(y, np.ndarray):
            return y
        elif isinstance(y, pd.Series):
            return y.values
        else:
            return y[target].values

    # def fit_scalers(self):
    #     """
    #     Fits both X and y scalers on the training data.
    #     Should be called after load_data() and before any transformations.
    #     """
    #     if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
    #         raise ValueError("Data must be loaded before fitting scalers. Call load_data() first.")
    #
    #     print("Fitting scalers on training data...")
    #
    #     # Fit X scaler
    #     if self.X_scaler is not None:
    #         self.X_scaler.fit(self.X_train)
    #
    #     # Fit y scaler
    #     if self.y_scaler is not None:
    #         # Reshape y_train to 2D array if it's 1D
    #         y_train_reshaped = self.y_train.reshape(-1, 1) if len(self.y_train.shape) == 1 else self.y_train
    #         self.y_scaler.fit(y_train_reshaped)
    #
    #     print("Scalers fitted successfully.")

    def get_feature_importances(self):
        """Get feature importances for both single and multi-output models."""
        if self.is_multi_output:
            all_importances = []
            for model in self.model.estimators_:
                all_importances.append(model.feature_importances_)
            return np.mean(all_importances, axis=0)
        else:
            return self.model.feature_importances_

    def get_feature_names(self):
        """Return feature names for both single and multi-output models."""
        if self.is_multi_output:
            first_estimator = next(iter(self.model.estimators_))
            return first_estimator.feature_names_in_
        else:
            return self.model.feature_names_in_

    def load_data(self, train_path, val_path, test_path):
        self.train_data = pd.read_parquet(train_path)
        self.val_data = pd.read_parquet(val_path)
        self.test_data = pd.read_parquet(test_path)

    def get_param_grid(self):
        # Define the hyperparameter grid with a maximum of 100 configurations
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "n_estimators": [20, 50, 100],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_lambda": [1, 2]
        }
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return random.sample(configurations, 100)

    def is_scaled(self, y):
        if np.all(y > 0):
            return False
        else:
            return True

    def inverse_scale_if_needed(self, y):
        if self.is_scaled(y):
            y = np.array(y).reshape(-1, 1)
            y = self.y_scaler.inverse_transform(y)
            y = y.ravel()
        return y

    def scale_y_values(self, y, y_pred):
        if not self.model_name.endswith('_plsr'):
            # For regular XGBoost, we need to scale both predictions and true values
            if isinstance(y, pd.DataFrame):
                # Extract only the target column if it's a DataFrame
                y_true = y[self.target_variables[0]].values if not self.is_multi_output else y[
                    self.target_variables].values
            else:
                # Convert Series to numpy array if needed
                y_true = y.values if isinstance(y, pd.Series) else y

            # Scale the true values
            y_true_scaled = self.y_scaler.transform(y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true).ravel()

            # Scale the predictions
            y_pred_scaled = y_pred.reshape(-1, 1)
            y_pred_scaled = self.y_scaler.transform(y_pred_scaled).ravel()

            # Use scaled values for metric calculation
            y_true_final = y_true_scaled
            y_pred_final = y_pred_scaled
        else:
            # For PLSR models, use as is
            if isinstance(y, pd.DataFrame):
                y_true_final = y[self.target_variables[0]].values if not self.is_multi_output else y[
                    self.target_variables].values
            else:
                y_true_final = y
            y_pred_final = y_pred
        return y_true_final, y_pred_final

    def train_and_evaluate_by_rmse_per_configuration(self, params, X_train, y_train, X_val, y_val):
        model_params = {k: v for k, v in params.items() if k not in ['eval_set', 'eval_metric', 'verbose']}

        if self.is_multi_output:
            model = MultiOutputRegressor(xgb.XGBRegressor(**model_params, eval_metric='rmse'))
        else:
            model = xgb.XGBRegressor(**model_params, eval_metric='rmse')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # y_true_final, y_pred_final = self.scale_y_values(y_val, y_pred)
        y_val = self.inverse_scale_if_needed(y_val)
        y_pred = self.inverse_scale_if_needed(y_pred)

        if self.is_multi_output:
            multi_rmses = np.sqrt(mean_squared_error(y_val, y_pred, multioutput='raw_values'))
            mean_rmses = np.mean(multi_rmses)
            return mean_rmses, multi_rmses.tolist()
        else:
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse, [rmse]

    def find_best_configuration_based_rmse_score(self, X_train, y_train, X_val, y_val, model_name):
        # find the best hyperparameters for each target variable and the mean
        param_grid = self.get_param_grid()

        # Initialize the best RMSE and parameters storage
        minimal_rmse = float("inf")
        best_params = {}

        # Initialize best_targets_rmses based on the model type
        if self.is_multi_output:
            best_targets_rmses = {target: None for target in TARGET_VARIABLES}
        else:
            best_targets_rmses = {self.target_variables[0]: None}

        # # todo return this code!!!!!-------------------------------------------------------------
        # Hyperparameter tuning loop
        for params in tqdm(param_grid, desc="Hyperparameter tuning"):
            rmse, multi_targets_rmses = self.train_and_evaluate_by_rmse_per_configuration(params, X_train, y_train,
                                                                                          X_val, y_val)
            # Update if we found better parameters
            if rmse < minimal_rmse:
                minimal_rmse = rmse
                best_params = params
                if self.is_multi_output:
                    best_targets_rmses = {target: multi_targets_rmses[i] for i, target in enumerate(TARGET_VARIABLES)}
                else:
                    best_targets_rmses = {self.target_variables[0]: multi_targets_rmses[0]}

        self.best_params = best_params
        # # todo return this code!!!!!-------------------------------------------------------------
        # self.best_params = {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.8,
        #                     'colsample_bytree': 0.8, 'gamma': 0, 'reg_lambda': 1}
        # best_targets_rmses = {'N_Value': np.float64(0.24239717956065013)}
        # minimal_rmse = np.float64(0.24239717956065013)
        best_targets_rmses[MEAN] = minimal_rmse
        self.targets_rmses_for_best_params = best_targets_rmses
        print(f"\nBest Configurations for {model_name} raised from Hyperparameter tuning:")
        print(self.best_params)

        return self.targets_rmses_for_best_params, self.best_params, minimal_rmse

    def k_fold_cross_validate_model(self, X_train, y_train, param_grid, y_scaler):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.train_rmses = {}
        self.val_rmses = {}
        self.scalers = {}

        is_multi_output = len(self.target_variables) > 1
        # todo - I don't see why does we nee ro do it if I have y_scaler from the PLSR package
        # # Fit scaler per target
        # for i, target in enumerate(self.target_variables):
        #     scaler = StandardScaler()
        #     if isinstance(y_train, np.ndarray):
        #         y_col = y_train[:, i].reshape(-1, 1)
        #     elif isinstance(y_train, pd.Series):
        #         y_col = y_train.values.reshape(-1, 1)
        #     else:
        #         y_col = y_train[target].values.reshape(-1, 1)
        #     scaler.fit(y_col)
        #     self.scalers[target] = scaler

        # self.y_scaler = self.scalers[self.target_variables[0]]  # default scaler

        params = self.best_params if hasattr(self, "best_params") else param_grid

        fold_train_final_rmse = {target: [] for target in self.target_variables}
        fold_val_final_rmse = {target: [] for target in self.target_variables}
        per_fold_train_rmses = {target: [] for target in self.target_variables}
        per_fold_val_rmses = {target: [] for target in self.target_variables}

        for train_index, val_index in tqdm(kf.split(X_train), desc="Cross-validation"):
            X_train_fold = X_train[train_index] if isinstance(X_train, np.ndarray) else X_train.iloc[train_index]
            X_val_fold = X_train[val_index] if isinstance(X_train, np.ndarray) else X_train.iloc[val_index]
            y_train_fold = y_train[train_index] if isinstance(y_train, np.ndarray) else y_train.iloc[train_index]
            y_val_fold = y_train[val_index] if isinstance(y_train, np.ndarray) else y_train.iloc[val_index]

            if is_multi_output:
                model = MultiOutputRegressor(xgb.XGBRegressor(**params, eval_metric='rmse'))
                model.fit(X_train_fold, y_train_fold)

                for i, estimator in enumerate(model.estimators_):
                    target = self.target_variables[i]
                    y_train_true = y_train_fold[:, i] if isinstance(y_train_fold, np.ndarray) else y_train_fold.iloc[:,
                                                                                                   i]
                    y_val_true = y_val_fold[:, i] if isinstance(y_val_fold, np.ndarray) else y_val_fold.iloc[:, i]

                    per_iter_train_rmse = []
                    per_iter_val_rmse = []

                    for round_num in range(1, estimator.best_iteration + 1 if hasattr(estimator,
                                                                                      "best_iteration") else estimator.n_estimators + 1):
                        y_train_pred_i = estimator.predict(X_train_fold, iteration_range=(0, round_num))
                        y_val_pred_i = estimator.predict(X_val_fold, iteration_range=(0, round_num))

                        # y_train_scaled = self.scalers[target].transform(y_train_true.values.reshape(-1, 1))
                        # y_train_pred_scaled = self.scalers[target].transform(y_train_pred_i.reshape(-1, 1))
                        # y_val_scaled = self.scalers[target].transform(y_val_true.values.reshape(-1, 1))
                        # y_val_pred_scaled = self.scalers[target].transform(y_val_pred_i.reshape(-1, 1))

                        rmse_train = mean_squared_error(y_train_true, y_train_pred_i, squared=False)
                        rmse_val = mean_squared_error(y_val_true, y_val_pred_i, squared=False)

                        per_iter_train_rmse.append(rmse_train)
                        per_iter_val_rmse.append(rmse_val)

                    per_fold_train_rmses[target].append(per_iter_train_rmse)
                    per_fold_val_rmses[target].append(per_iter_val_rmse)

                    # Final RMSE
                    final_train_rmse = per_iter_train_rmse[-1]
                    final_val_rmse = per_iter_val_rmse[-1]
                    fold_train_final_rmse[target].append(final_train_rmse)
                    fold_val_final_rmse[target].append(final_val_rmse)

            else:
                target = self.target_variables[0]
                model = xgb.XGBRegressor(**params, eval_metric='rmse')
                y_train_target = self.extract_target_values(y_train_fold, target)
                y_val_target = self.extract_target_values(y_val_fold, target)

                model.fit(
                    X_train_fold,
                    y_train_target,
                    eval_set=[(X_train_fold, y_train_target), (X_val_fold, y_val_target)],
                    verbose=False
                )

                per_iter_train_rmse = []
                per_iter_val_rmse = []

                for round_num in range(1, model.best_iteration + 1 if hasattr(model,
                                                                              "best_iteration") else model.n_estimators + 1):
                    y_train_pred_i = model.predict(X_train_fold, iteration_range=(0, round_num))
                    y_val_pred_i = model.predict(X_val_fold, iteration_range=(0, round_num))

                    # y_train_true_scaled, y_train_pred_scaled = self.scale_y_values(y_train_fold, y_train_pred_i)
                    # y_val_true_scaled, y_val_pred_scaled = self.scale_y_values(y_val_fold, y_val_pred_i)

                    rmse_train = np.sqrt(mean_squared_error(y_train_fold, y_train_pred_i))
                    rmse_val = np.sqrt(mean_squared_error(y_val_fold, y_val_pred_i))

                    per_iter_train_rmse.append(rmse_train)
                    per_iter_val_rmse.append(rmse_val)

                per_fold_train_rmses[target].append(per_iter_train_rmse)
                per_fold_val_rmses[target].append(per_iter_val_rmse)

                # Final RMSE
                final_train_rmse = per_iter_train_rmse[-1]
                final_val_rmse = per_iter_val_rmse[-1]
                fold_train_final_rmse[target].append(final_train_rmse)
                fold_val_final_rmse[target].append(final_val_rmse)

            self.model = model  # Save last model

        # Aggregate per-iteration RMSEs
        for target in self.target_variables:
            self.train_rmses[target] = np.mean(per_fold_train_rmses[target], axis=0)
            self.val_rmses[target] = np.mean(per_fold_val_rmses[target], axis=0)
            self.final_train_rmse[target] = np.mean(fold_train_final_rmse[target])
            self.final_val_rmse[target] = np.mean(fold_val_final_rmse[target])

    def evaluate_model(self, X, y, model, dataset_type='validation'):
        # Get predictions
        y_pred = model.predict(X)

        y = self.inverse_scale_if_needed(y)
        y_pred = self.inverse_scale_if_needed(y_pred)
        # y_true_final, y_pred_final = self.scale_y_values(y, y_pred)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Calculate adjusted R²
        n = len(y)
        p = X.shape[1]
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Store metrics
        metrics_dict = {
            'rmse': rmse,
            'r2': r2,
            'r2_adj': r2_adj
        }

        if dataset_type == 'validation':
            self.evaluated_val_rmses.update(metrics_dict)
        else:  # test
            self.evaluated_test_rmses.update(metrics_dict)

        print(f"\n{dataset_type.capitalize()} Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Adjusted R²: {r2_adj:.4f}")

        return y_pred

    def save_model_object(self, data_folder_spec):
        print("Saving the model object after evaluation...")
        directory = os.path.join(self.save_dir, data_folder_spec, self.model_name)
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, 'model.pkl'))

    def run(self, train_path, val_path, test_path, data_folder_spec, scaled=False):
        print(f"Running {self.model_name} XGBoost{'MultiOutput' if self.is_multi_output else ''}")

        self.load_data(train_path, val_path, test_path)
        X_train, y_train = self.preprocess_data(dataset="train")
        X_val, y_val = self.preprocess_data(dataset="val")
        X_test, y_test = self.preprocess_data(dataset="test")

        # todo - set as irrelevant today - I'll inverse at the end
        # if self.model_name.endswith('_plsr'):
        #     self.y_scaler.fit(y_train)
        #     y_val = self.y_scaler.inverse_transform(y_val)
        #     y_test = self.y_scaler.inverse_transform(y_test)

        #     # Transform validation and test data using the fitted scaler
        #     if not self.is_multi_output:
        #         y_val = self.y_scaler.transform(
        #             y_val.values.reshape(-1, 1) if isinstance(y_val, pd.Series) else y_val.reshape(-1, 1))
        #         y_test = self.y_scaler.transform(
        #             y_test.values.reshape(-1, 1) if isinstance(y_test, pd.Series) else y_test.reshape(-1, 1))
        #     else:
        #         y_val = self.y_scaler.transform(y_val)
        #         y_test = self.y_scaler.transform(y_test)
        # # Fit the y_scaler on training data if not already fitted
        # if not self.model_name.endswith('_plsr'):
        #     if not self.is_multi_output:
        #         y_train_reshaped = y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train.reshape(
        #             -1, 1)
        #     else:
        #         y_train_reshaped = y_train
        #     # Fit and transform the training data
        #     self.y_scaler.fit(y_train_reshaped)
        #     y_train = self.y_scaler.transform(y_train_reshaped)
        #
        #     # Transform validation and test data using the fitted scaler
        #     if not self.is_multi_output:
        #         y_val = self.y_scaler.transform(
        #             y_val.values.reshape(-1, 1) if isinstance(y_val, pd.Series) else y_val.reshape(-1, 1))
        #         y_test = self.y_scaler.transform(
        #             y_test.values.reshape(-1, 1) if isinstance(y_test, pd.Series) else y_test.reshape(-1, 1))
        #     else:
        #         y_val = self.y_scaler.transform(y_val)
        #         y_test = self.y_scaler.transform(y_test)

        print("Finding best configurations...")
        best_rmses, best_params, best_multi_rmses = self.find_best_configuration_based_rmse_score(
            X_train, y_train, X_val, y_val, self.model_name)
        print("\nMinimal RMSEs based on the chosen configuration:")
        for key, value in best_rmses.items():
            print(f"{key}: {value}")

        self.k_fold_cross_validate_model(X_train, y_train, best_params, self.y_scaler)
        self.evaluate_model(X_val, y_val, self.model, dataset_type='validation')
        self.save_model_object(data_folder_spec)
        self.evaluate_model(X_test, y_test, self.model, dataset_type='test')
