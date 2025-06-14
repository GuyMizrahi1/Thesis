from itertools import product
from sklearn.model_selection import KFold
from constants_config import TARGET_VARIABLES
from tqdm import tqdm

import numpy as np


def trainingInit(model_obj):
    """
    model_obj: a baseModel Object (can be PLSR, XGBoost and RandomForest)
    """

    if model_obj.is_multi_output and model_obj.target_variable_name is None:
        rmse_results = {target_var: [] for target_var in TARGET_VARIABLES}
        rmse_results['Avg_RMSE'] = []
    else:
        rmse_results = {model_obj.target_variable_name: []}

    return rmse_results


def createHyperParametersCombinations(param_grid):
    # This function creates a hyperparameter combination - easier to loop on.
    values = param_grid.values()
    hyper_parameters_names = param_grid.keys()
    param_combos = {}
    combinations = list(product(*values))

    for i, comb in tqdm(enumerate(combinations), total=len(combinations),
                        desc='Creating Hyperparameter Combinations'):
        hyper_param = dict(zip(hyper_parameters_names, comb))
        param_combos[f'Combination_{i}'] = hyper_param

    return param_combos


def tuningLoop(model_obj, params, PLSR_Tuning):
    rmse_results = trainingInit(model_obj)
    if model_obj.is_multi_output:
        n_value, sc_value, st_value = TARGET_VARIABLES

    if PLSR_Tuning:
        for n_components in tqdm(params, desc='Optimizing Number of Components', total=len(params)):
            if model_obj.is_multi_output:
                model_obj.model.estimator.set_params(n_components=n_components)
            else:
                model_obj.model.set_params(n_components=n_components)
            rmse = model_obj.validate()

            if model_obj.is_multi_output:
                sc_value_rmse, st_value_rmse = rmse # todo vine
                # n_value_rmse, sc_value_rmse, st_value_rmse = rmse
                avg_rmse = np.mean(rmse)
                # Mainly For plotting
                # rmse_results[n_value] += [n_value_rmse]  todo vine
                rmse_results[sc_value] += [sc_value_rmse]
                rmse_results[st_value] += [st_value_rmse]
                # For choosing the best hyperparameters
                rmse_results['Avg_RMSE'] += [(model_obj.model.estimator.get_params(), avg_rmse)]
            else:
                rmse_results[model_obj.target_variable_name] += [(model_obj.model.get_params(), rmse)]

    else:

        for hyperparams in tqdm(params.values(), desc='Tuning Hyperparameters', total=len(params)):
            # hyperparams is a dictionary
            # combination is the number of the combination
            print(f'HyperParameters:{hyperparams}')
            if model_obj.is_multi_output:
                model_obj.model.estimator.set_params(**hyperparams)
            else:
                model_obj.model.set_params(**hyperparams)
            rmse = model_obj.validate()

            if model_obj.is_multi_output:
                # todo vine
                # n_value_rmse, sc_value_rmse, st_value_rmse = rmse
                sc_value_rmse, st_value_rmse = rmse
                avg_rmse = np.mean(rmse)
                # Mainly For plotting
                # todo vine
                # rmse_results[n_value] += [n_value_rmse]
                rmse_results[sc_value] += [sc_value_rmse]
                rmse_results[st_value] += [st_value_rmse]
                # For choosing the best hyperparameters
                rmse_results['Avg_RMSE'] += [(model_obj.model.estimator.get_params(), avg_rmse)]

            else:
                rmse_results[model_obj.target_variable_name] += [(model_obj.model.get_params(), rmse)]
    return rmse_results


# Use this function for tuning your model's hyperparameters.
def hyperParameterTuning(model_obj, PLSR_Tuning=False):
    """
    model: a baseModel Object (can be PLSR, XGBoost and RandomForest)
    target_variable_name: the name of the target_variable (if not using multioutput.)
    """

    if not PLSR_Tuning:
        # For random forest and XGBoost always create combinations
        params = createHyperParametersCombinations(model_obj.param_grid)
    else:
        # For PLSR dont create - only one hyperparameter
        params = model_obj.param_grid['n_components']

    rmse_results = tuningLoop(model_obj, params, PLSR_Tuning)

    return rmse_results


# Use this function after you've finished optimizing your model!
def CV10(model_obj, n_splits=10):
    """
    model: a baseModel Object (can be PLSR, XGBoost and RandomForest) - With the best Hyper Parameters.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_results = trainingInit(model_obj)

    if model_obj.is_multi_output:
        n_value, sc_value, st_value = TARGET_VARIABLES

    for train_index, val_index in tqdm(kf.split(model_obj.dataset.X_train), total=n_splits, desc='Cross Validation'):

        X_train_fold, X_val_fold = model_obj.dataset.X_train.iloc[train_index], model_obj.dataset.X_train.iloc[
            val_index]
        y_train_fold, y_val_fold = model_obj.dataset.Y_train.iloc[train_index], model_obj.dataset.Y_train.iloc[
            val_index]

        rmses = model_obj.CrossValidate(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        if model_obj.is_multi_output:
            # todo vine
            # n_rmse, sc_rmse, st_rmse = rmses
            # rmse_results[n_value] += [n_rmse]
            sc_rmse, st_rmse = rmses
            rmse_results[sc_value] += [sc_rmse]
            rmse_results[st_value] += [st_rmse]
            rmse_results['Avg_RMSE'] += [np.mean(rmses)]
        else:
            dependent_feature_rmse = rmses
            rmse_results[model_obj.target_variable_name] += [dependent_feature_rmse]

    return rmse_results
