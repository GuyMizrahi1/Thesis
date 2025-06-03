import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_based_prediction.constants_config import data_folder_spec


class BaseModel:

    def __init__(self, dataset, model, param_grid, is_multi_output=False, target_variable_name=None):

        self.dataset = dataset.copy()
        self.target_variable_name = target_variable_name

        if self.target_variable_name:
            self.dataset.Y_train = self.dataset.Y_train[self.target_variable_name]
            self.dataset.Y_val = self.dataset.Y_val[self.target_variable_name]
            self.dataset.Y_test = self.dataset.Y_test[self.target_variable_name]

        self.model = model
        self.is_multi_output = is_multi_output
        self.param_grid = param_grid
        self.best_params = None
        self.y_scaler = None

        self.random_state = 42

        self.evaluation_metrics = None

    def load_scaler(self, scaler_path):
        self.y_scaler = joblib.load(scaler_path)

    # def inverse_transform(self, y):
    #     os.path.join('./models', self.dataset.folder_name, 'y_scaler.pkl')
    #     if self.y_scaler:
    #         return self.y_scaler.inverse_transform(y)
    #     return y

    def inverse_transform(self, y):
        if self.y_scaler is None:
            print("Debug: Loading scaler...")
            # Get the current working directory
            current_dir = os.getcwd()

            # Remove a duplicated path if we're already in the PLSR directory
            if current_dir.endswith('spectral_based_prediction/PLSR'):
                primary_path = os.path.join(current_dir, 'models', data_folder_spec, 'y_scaler.pkl')
            else:
                base_dir = os.path.dirname(os.path.dirname(current_dir))
                primary_path = os.path.join(base_dir, 'spectral_based_prediction', 'PLSR', 'models',
                                            data_folder_spec, 'y_scaler.pkl')

            try:
                self.y_scaler = joblib.load(primary_path)
                print(f"Debug: Scaler loaded successfully from {primary_path}")
            except FileNotFoundError:
                print(f"Debug: Scaler not found at {primary_path}")
                return y

        return self.y_scaler.inverse_transform(y)

    def computeRMSE(self, Y, Y_hat):
        """
        Just a method to compute RMSE
        """

        if self.is_multi_output:
            n_value_rmse, sc_value_rmse, st_value_rmse = np.sqrt(
                mean_squared_error(y_true=Y, y_pred=Y_hat, multioutput='raw_values'))
            return n_value_rmse, sc_value_rmse, st_value_rmse

        return np.sqrt(mean_squared_error(y_true=Y, y_pred=Y_hat))

    def CrossValidate(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train)
        y_hat = self.model.predict(x_val)

        if self.is_multi_output:
            n_value_rmse, sc_value_rmse, st_value_rmse = self.computeRMSE(y_val, y_hat)
            return n_value_rmse, sc_value_rmse, st_value_rmse

        return self.computeRMSE(y_val, y_hat)

    def validate(self):
        """
        Can be used for Hyperparameter Tuning.
        """
        self.model.fit(self.dataset.X_train, self.dataset.Y_train)
        y_val = self.dataset.Y_val.squeeze()
        y_hat = self.model.predict(self.dataset.X_val).squeeze()
        # Convert to a numpy array and ensure 2D shape for inverse transform
        if y_hat.ndim == 1:
            y_hat = np.array(y_hat).reshape(-1, 1)
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy().reshape(-1, 1)
        elif y_val.ndim == 1:
            y_val = np.array(y_val).reshape(-1, 1)

        y_val_inv = self.inverse_transform(y_val)
        y_hat_inv = self.inverse_transform(y_hat)

        return self.computeRMSE(y_val_inv, y_hat_inv)

    def calculate_metrics(self, y_true, y_pred, X_test):
        print(f"Debug: Metrics calculation")
        print(f"Debug: y_true shape: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}, y_pred shape: "
              f"{y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred)}")
        print(f"Debug: y_true range: [{np.min(y_true)}, {np.max(y_true)}]")
        print(f"Debug: y_pred range: [{np.min(y_pred)}, {np.max(y_pred)}]")

        # Convert pandas Series to numpy array if necessary
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()

        # Ensure both arrays are 2D
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        n = len(y_true)
        p = 1  # number of predictors

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        print(f"Debug: RMSE: {rmse}")
        print(f"Debug: R2: {r2}")
        print(f"Debug: R2 Adjusted: {r2_adj}")

        return {
            'rmse': rmse,
            'r2': r2,
            'r2_adj': r2_adj
        }

    def calculate_rmsecv(self, X, y, cv=10):
        """Calculate Root Mean Square Error of Cross-Validation."""
        if self.is_multi_output:
            rmsecv = {}
            for i, target in enumerate(self.target_variable_name):
                mse_scores = cross_val_score(
                    self.model,
                    X,
                    y[:, i],
                    scoring='neg_mean_squared_error',
                    cv=cv
                )
                rmsecv[target] = np.sqrt(-mse_scores.mean())
        else:
            mse_scores = cross_val_score(
                self.model,
                X,
                y,
                scoring='neg_mean_squared_error',
                cv=cv
            )
            rmsecv = {self.target_variable_name: np.sqrt(-mse_scores.mean())}

        return rmsecv

    def calculate_vip_scores(self, X):
        if hasattr(self.model, 'feature_importances_'):
            # For RF and XGBoost
            return {
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
            }
        elif hasattr(self.model, 'coef_'):
            # For PLSR
            try:
                # Get PLSR specific attributes
                t = self.model.x_scores_
                w = self.model.x_weights_
                q = self.model.y_loadings_
                p = w.shape[0]  # number of variables
                h = w.shape[1]  # number of components

                # Calculate VIP scores
                vips = np.zeros(p)
                s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
                total_s = np.sum(s)

                for i in range(p):
                    weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
                    vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

                return {
                    'vip_scores': dict(zip(X.columns, vips))
                }
            except AttributeError:
                return {'vip_scores': None}
        return {'vip_scores': None}

    def evaluate(self):
        y_pred = self.model.predict(self.dataset.X_test)
        y_true = self.dataset.Y_test

        metrics = self.calculate_metrics(y_true, y_pred, self.dataset.X_test)
        rmsecv = self.calculate_rmsecv(self.dataset.X_train, self.dataset.Y_train)
        vip = self.calculate_vip_scores(self.dataset.X_train)

        # Store all metrics
        self.evaluation_metrics = {
            'test_metrics': metrics,
            'rmsecv': rmsecv,
            'best_n_components': getattr(self, "best_n_components", None),
            'variable_importance': vip
        }

        return metrics


def r2_adjusted_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
