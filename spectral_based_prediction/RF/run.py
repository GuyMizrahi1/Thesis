import os
from RF import RFModel
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

    # Initialize Dataset
    dataset = Dataset(
        train_file_name=f'test_data_{n_components}_plsr.parquet',
        validation_file_name=f'validation_data_{n_components}_plsr.parquet',
        test_file_name=f'test_data_{n_components}_plsr.parquet',
        data_folder_spec=data_folder_spec,
        target_variables=target_variables

    )

    # Initialize and run RFModel
    if len(target_variables) > 1:
        rf_model = RFModel(dataset, param_grid, MODEL_PATH)
    else:
        rf_model = RFModel(dataset, param_grid, MODEL_PATH, is_multi_output=False, target_variable_name=target_variables[0])

    if TRAINING_MODE:
        rf_model.run()
    else:
        rf_model = rf_model.load_model()
        rf_model.eval_plot()
