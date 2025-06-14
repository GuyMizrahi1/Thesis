import os
import math
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from baseline_for_training.Dataset import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from baseline_for_training.baseModels import BaseModel
from spectral_based_prediction.constants_config import AVG_RMSE
from baseline_for_training.Training_and_Tuning import hyperParameterTuning, CV10


class RFModel(BaseModel):
    def __init__(self, dataset: Dataset, param_grid: dict, model_path: str, is_multi_output=True,
                 target_variable_name=None):
        if is_multi_output:
            model = MultiOutputRegressor(RandomForestRegressor())
        else:
            model = RandomForestRegressor()
        super().__init__(dataset, model, param_grid, is_multi_output, target_variable_name)
        self.best_params = None
        self.cv_rmse = None
        self.test_rmse = {}
        self.model_path = model_path
        self.target_names = dataset.Y_train.columns.tolist()

    def run(self):
        self.__tune_hyperparameters()
        self.__cross_validate()
        self.__train_and_evaluate_model()
        self.eval_plot()
        self.__save_test_results()
        self.__save_model()

    def __tune_hyperparameters(self):
        print("Starting hyperparameter tuning...")
        tuning_results = hyperParameterTuning(self, PLSR_Tuning=False)

        if self.is_multi_output:
            # For multiple targets, use the first target's results
            metric_key = AVG_RMSE
            self.best_params = min(tuning_results[metric_key], key=lambda x: x[1])[0]
            self.model.estimator.set_params(**self.best_params)
        else:
            # For a single target
            metric_key = self.target_variable_name
            self.best_params = min(tuning_results[metric_key], key=lambda x: x[1])[0]
            self.model.set_params(**self.best_params)

        print(f"Best Parameters: {self.best_params}")

    def __cross_validate(self):
        print("Starting cross-validation...")
        self.cv_rmse = CV10(self, n_splits=10)

        # Plot RMSE vs Folds
        plt.figure(figsize=(10, 6))
        for target in self.target_names:
            plt.plot([i for i in range(1, 11)], self.cv_rmse[target], label=target)

        plt.xlabel('Folds')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Folds')
        plt.legend()
        plt.savefig(os.path.join(self.model_path, 'RMSE_vs_Folds_RF.png'))
        plt.close()

    def __train_and_evaluate_model(self):
        print("Training on train set...")
        self.model.fit(self.dataset.X_train, self.dataset.Y_train)
        print("Evaluating on test set...")
        rmse_results = self.evaluate()

        if self.is_multi_output:
            self.test_rmse.update(zip(self.target_names, rmse_results))
        else:
            # Handle a single output case
            self.test_rmse[self.target_variable_name] = rmse_results

        print(f"Test RMSE: {self.test_rmse}")

    def eval_plot(self):
        if isinstance(self.dataset.Y_test, pd.Series):  # Convert Series to DataFrame
            self.dataset.Y_test = self.dataset.Y_test.to_frame()

        # Plot predicted vs actual values
        y_pred_scaled = self.model.predict(self.dataset.X_test)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        # Convert predicted values back to their original scale
        y_hat = pd.DataFrame(
            self.dataset.Y_scaler.inverse_transform(y_pred_scaled),
            columns=self.dataset.Y_test.columns
        )

        y_test = self.dataset.Y_test_non_scaled

        n_targets = len(self.target_names)
        n_cols = min(2, n_targets)
        n_rows = math.ceil(n_targets / n_cols)

        # Predicted vs Actual plots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_targets == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)

        for idx, target in enumerate(self.target_names):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].scatter(y_test.iloc[:, idx].values, y_hat.iloc[:, idx].values)
            axs[row, col].set_title(target)
            axs[row, col].set_xlabel('Actual')
            axs[row, col].set_ylabel('Predicted')

        # Disable unused subplots
        for idx in range(n_targets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'Predicted_vs_Actual_Values_RF.png'))
        plt.close()

        # Residuals plots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_targets == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)

        residuals = y_test.values - y_hat.values

        for idx, target in enumerate(self.target_names):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].scatter(y_hat.values[:, idx], residuals[:, idx])  # Explicitly use `.values`
            axs[row, col].set_title(target)
            axs[row, col].set_xlabel('Predicted')
            axs[row, col].set_ylabel('Residual')
            axs[row, col].axhline(y=0, color='r', linestyle='-')

        # Disable unused subplots
        for idx in range(n_targets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'Residuals_vs_Predicted_Values_RF.png'))
        plt.close()

    def __save_model(self):
        print("Saving model...")
        joblib.dump(self, os.path.join(self.model_path, 'rf_model.pkl'))
        print("Model saved.")

    def __save_test_results(self):
        print("Saving test results...")
        with open(os.path.join(self.model_path, 'test_rmse_results.json'), 'w') as f:
            json.dump(self.test_rmse, f)

    def load_model(self):
        print("Loading model...")
        model = joblib.load(os.path.join(self.model_path, 'rf_model.pkl'))
        print("Model loaded")
        return model
