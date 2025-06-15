import joblib
import argparse
import os.path
from spectral_based_prediction.XGBoost.xgboost_generic_implement import XGBoostGeneric
from spectral_based_prediction.constants_config import DATA_FOLDER_PATH, FIGURE_FOLDER_PATH, TARGET_VARIABLES, \
    data_folder_spec, target_variables, n_components
from utils import plot_chosen_configurations_rmse, load_model, ensure_data_paths_exist, plot_learning_curves, \
    plot_feature_importances, plot_residuals, save_test_scores


def main(train: bool = True, single_target: bool = True, target: str = target_variables[0]):
    parser = argparse.ArgumentParser(description="Train or load XGBoost model")
    args = parser.parse_args()
    plsr_comp = n_components

    # Ensure the directory exists and get data paths
    train_path, val_path, test_path, train_plsr_path, val_plsr_path, test_plsr_path = ensure_data_paths_exist(
        DATA_FOLDER_PATH, data_folder_spec=data_folder_spec, plsr_comp=f'{plsr_comp}')

    # Load the y_scaler saved from PLSR at the beginning
    base_dir = os.path.dirname(os.getcwd())
    y_scaler_path = os.path.join(base_dir, 'PLSR', 'models', data_folder_spec, 'y_scaler.pkl')
    y_scaler = joblib.load(y_scaler_path)

    # Configure models based on a single / multi-target setting
    if single_target:
        if not target:
            raise ValueError("Must specify --target when using --single-target")

        # Create single-target models
        xgb_model = XGBoostGeneric(
            model_name='xgboost_single_target',
            is_multi_output=False,
            target_variables=target,
            y_scaler=y_scaler
        )
        xgb_model_plsr = XGBoostGeneric(
            model_name='xgboost_single_target_plsr',
            is_multi_output=False,
            target_variables=target,
            y_scaler=y_scaler
        )
    else:
        # Create multi-target models
        xgb_model = XGBoostGeneric(
            model_name='xgboost_multi_target',
            is_multi_output=True,
            target_variables=target_variables,
            y_scaler=y_scaler
        )
        xgb_model_plsr = XGBoostGeneric(
            model_name='xgboost_multi_target_plsr',
            is_multi_output=True,
            target_variables=target_variables,
            y_scaler=y_scaler
        )

    if train or args.train:
        # run both XGBoost models
        xgb_model.run(train_path, val_path, test_path, data_folder_spec)
        xgb_model_plsr.run(train_plsr_path, val_plsr_path, test_plsr_path, data_folder_spec, True)
    else:
        model_suffix = 'single_target' if args.single_target else 'multi_target'
        xgb_model = load_model(directory=f"models/xgboost_{model_suffix}")
        xgb_model_plsr = load_model(directory=f"models/xgboost_{model_suffix}_plsr")

    figures_path = os.path.join(FIGURE_FOLDER_PATH, data_folder_spec)
    # Create plots
    plot_chosen_configurations_rmse(xgb_model, xgb_model_plsr, single_target, figures_path)
    plot_learning_curves(xgb_model, xgb_model_plsr, figures_path)
    plot_feature_importances(xgb_model, xgb_model_plsr, figures_path)
    if single_target:
        plot_residuals(xgb_model, xgb_model_plsr, single_target, target, test_path, test_plsr_path, figures_path)
    else:
        for target in target_variables:
            plot_residuals(xgb_model, xgb_model_plsr, single_target, target, test_path, test_plsr_path, figures_path)
    # Save test scores
    save_test_scores(xgb_model, xgb_model_plsr, single_target, test_path, test_plsr_path, figures_path)


if __name__ == "__main__":
    main(single_target=False, target=target_variables)
