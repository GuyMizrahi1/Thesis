import os
import joblib
# import pandas as pd
from RF import RFModel
from sklearn.preprocessing import StandardScaler
from baseline_for_training.Dataset import Dataset
from spectral_based_prediction.constants_config import data_folder_spec, target_variables, n_components

if __name__ == '__main__':

    # Set Training Mode
    TRAINING_MODE = True
    MODEL_PATH = f'./models/{data_folder_spec}'

    # Create the model directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)

    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize Dataset with y_scaler=False, no target scaling applied
    dataset = Dataset(
        train_file_name=f'test_data_{n_components}_plsr.parquet',
        validation_file_name=f'validation_data_{n_components}_plsr.parquet',
        test_file_name=f'test_data_{n_components}_plsr.parquet',
        data_folder_spec=data_folder_spec,
        target_variables=target_variables
    )

    # Dynamically generate scalers if they do not exist
    base_dir = os.path.dirname(os.getcwd())  # This gets the current working directory (run.py location)
    x_scaler_path = os.path.join(base_dir, 'PLSR', 'models', data_folder_spec, 'X_scaler.pkl')
    y_scaler_path = os.path.join(base_dir, 'PLSR', 'models', data_folder_spec, 'Y_scaler.pkl')
    # # todo - I think that this part is irrelevant, I already have the scaler from the plsr
    # if os.path.exists(x_scaler_path):
    #     X_scaler = joblib.load(x_scaler_path)
    # else:
    #     # Generate a new X_scaler from training data and save it
    #     X_scaler = StandardScaler()
    #     X_scaler.fit(dataset.X_train.values)
    #     joblib.dump(X_scaler, x_scaler_path)
    #
    # if os.path.exists(y_scaler_path):
    #     Y_scaler = joblib.load(y_scaler_path)
    # else:
    #     # Generate a new Y_scaler from training data and save it
    #     Y_scaler = StandardScaler()
    #     Y_scaler.fit(dataset.Y_train.values)
    #     joblib.dump(Y_scaler, y_scaler_path)
    # # todo - Until here
    # # Preprocessing Data
    # dataset.X_train[dataset.X_train.columns] = X_scaler.transform(dataset.X_train.values)
    # dataset.Y_train[dataset.Y_train.columns] = Y_scaler.transform(dataset.Y_train.values)
    #
    # dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
    # dataset.Y_val[dataset.Y_val.columns] = Y_scaler.transform(dataset.Y_val.values)
    #
    # dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)
    # dataset.Y_test[dataset.Y_test.columns] = Y_scaler.transform(dataset.Y_test.values)

    # Attach scalers to the dataset instance
    dataset.X_scaler = joblib.load(x_scaler_path)
    dataset.Y_scaler = joblib.load(y_scaler_path)

    # Initialize and run RFModel
    if len(target_variables) > 1:
        rf_model = RFModel(dataset, param_grid, MODEL_PATH)
    else:
        rf_model = RFModel(dataset, param_grid, MODEL_PATH, is_multi_output=False,
                           target_variable_name=target_variables[0])

    if TRAINING_MODE:
        rf_model.run()
    else:
        rf_model = rf_model.load_model()
        rf_model.eval_plot()
