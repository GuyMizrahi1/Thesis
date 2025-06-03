import json
import os.path
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PLSR_class import PLSRModel
from spectral_based_prediction.baseline_for_training.Dataset import Dataset
from spectral_based_prediction.baseline_for_training.Training_and_Tuning import CV10
from spectral_based_prediction.baseline_for_training.Training_and_Tuning import hyperParameterTuning
from spectral_based_prediction.constants_config import data_folder_spec, target_variables

def create_json_file(rmses, path, filename):
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(rmses, f)

def figures_for_multi_output_plsr():
    # Correlation matrix - dependent variables
    plt.figure(figsize=(8, 5))
    sns.heatmap(pd.concat([dataset.Y_train, dataset.Y_val, dataset.Y_test], axis=0).corr(), annot=True, fmt=".2f",
                cmap='Reds')
    plt.title('Correlation matrix for X and Y')
    plt.savefig('./Plots/Correlation_matrix.png')

    # Pair plot - dependent variables
    plt.figure(figsize=(8, 5))
    sns.pairplot(pd.concat([dataset.Y_train, dataset.Y_val, dataset.Y_test], axis=0))
    plt.savefig('./Plots/Pair_plot.png')


if __name__ == '__main__':
    # creating Dataset Instance
    train_file_name = 'train_data.parquet'
    validation_file_name = 'validation_data.parquet'
    test_file_name = 'test_data.parquet'

    dataset = Dataset(train_file_name, validation_file_name, test_file_name, data_folder_spec, target_variables)

    # Create PLSR models based on target_variables
    models = {}
    rmses = {}

    if len(target_variables) > 1:
        figures_for_multi_output_plsr()

    # Preprocessing Data
    X_scaler = joblib.load(os.path.join('./models', data_folder_spec, 'X_scaler.pkl'))
    Y_scaler = joblib.load(os.path.join('./models', data_folder_spec, 'y_scaler.pkl'))

    dataset.X_train[dataset.X_train.columns] = X_scaler.fit_transform(dataset.X_train.values)
    dataset.Y_train[dataset.Y_train.columns] = Y_scaler.fit_transform(dataset.Y_train.values)

    dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
    dataset.Y_val[dataset.Y_val.columns] = Y_scaler.transform(dataset.Y_val.values)

    dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)
    dataset.Y_test[dataset.Y_test.columns] = Y_scaler.transform(dataset.Y_test.values)

    # Preparing Models
    # Define the parameter grid
    param_grid = {'n_components': [i for i in range(1, 51)]}

    # Create a Multi-output model if there's more than one target
    if len(dataset.target_variables) > 1:
        multi_PLSR = PLSRModel(dataset, param_grid, is_multi_output=True)
        models['multi'] = multi_PLSR
        multi_rmses = hyperParameterTuning(multi_PLSR, PLSR_Tuning=True)
        rmses['multi'] = multi_rmses

    # Create individual models for each target
    for target in dataset.target_variables:
        model = PLSRModel(dataset, param_grid, is_multi_output=False, target_variable_name=target)
        models[target] = model
        target_rmses = hyperParameterTuning(model, PLSR_Tuning=True)
        rmses[target] = target_rmses

    path = f'./outputs/{data_folder_spec}'
    joblib.dump(Y_scaler, os.path.join('./models', data_folder_spec, 'y_scaler.pkl'))

    # Save results
    for model_name, model_rmses in rmses.items():
        create_json_file(model_rmses, path, f'{model_name}_rmses.json')

    # Get the best number of components for each model
    best_components = {}
    for model_name, model_rmses in rmses.items():
        if model_name == 'multi':
            best_n = sorted(model_rmses['Avg_RMSE'], key=lambda x: x[1])[0][0]['n_components']
        else:
            best_n = sorted(model_rmses[model_name], key=lambda x: x[1])[0][0]['n_components']
        best_components[model_name] = best_n
        print(f'Best {model_name} PLSR Number of Components:', best_n)

    # Training using CV10
    # Set the best components for each model
    for model_name, model in models.items():
        if model_name == 'multi':
            model.model.estimator.set_params(n_components=best_components[model_name])
        else:
            model.model.set_params(n_components=best_components[model_name])

        model.best_n_components = best_components[model_name]

    # Perform 10-fold cross-validation
    cv10_results = {}
    for model_name, model in models.items():
        cv10_rmse = CV10(model)
        cv10_results[model_name] = cv10_rmse
        create_json_file(cv10_rmse, path, f'{model_name}_rmse_cv10.json')

    # Fit models on all data for saving
    for model_name, model in models.items():
        if model_name == 'multi':
            model.model.estimator.fit(dataset.X_train, dataset.Y_train)
        else:
            target_idx = dataset.target_variables.index(model_name)
            model.model.fit(dataset.X_train, dataset.Y_train.iloc[:, target_idx])

    # Eval on test data and save results for each model
    for model_name, model in models.items():
        # Evaluate final model performance on testset, and collect RMSECV and VIP
        test_metrics = model.evaluate()
        print("Test Metrics (RMSE, R²):", test_metrics)
        print("RMSECV:", model.evaluation_metrics["rmsecv"])
        print('best_n_components', getattr(model, "best_n_components", None))
        print("VIP Scores:", model.evaluation_metrics["variable_importance"])
        create_json_file(model.evaluation_metrics, path ,f'{model_name}_rmse_test.json')

        # save the model with an appropriate name
        if model_name == 'multi':
            joblib.dump(model.model.estimator, f'./models/{data_folder_spec}/multi_plsr.pkl')
        else:
            joblib.dump(model.model, f'./models/{data_folder_spec}/{model_name}_plsr.pkl')

