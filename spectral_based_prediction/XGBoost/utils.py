import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from spectral_based_prediction.constants_config import COLOR_PALETTE_FOR_TARGET_VARIABLES, TARGET_VARIABLES_WITH_MEAN, \
    NON_FEATURE_COLUMNS, TARGET_VARIABLES, COLOR_PALETTE_FOR_TWO_MODELS, ColumnName


def ensure_data_paths_exist(data_folder_path, data_folder_spec: str, plsr_comp: str):
    """Ensure the directory exists and return paths for train, validation, and test data."""
    os.makedirs(data_folder_path, exist_ok=True)

    data_folder_path = os.path.join(data_folder_path, data_folder_spec)
    train_path = os.path.join(data_folder_path, 'train_data.parquet')
    val_path = os.path.join(data_folder_path, 'validation_data.parquet')
    test_path = os.path.join(data_folder_path, 'test_data.parquet')
    train_plsr_path = os.path.join(data_folder_path, f'train_data_{plsr_comp}_plsr.parquet')
    val_plsr_path = os.path.join(data_folder_path, f'validation_data_{plsr_comp}_plsr.parquet')
    test_plsr_path = os.path.join(data_folder_path, f'test_data_{plsr_comp}_plsr.parquet')

    return train_path, val_path, test_path, train_plsr_path, val_plsr_path, test_plsr_path


def load_model(directory):
    return joblib.load(os.path.join(directory, "model.pkl"))


def plot_chosen_configurations_rmse(model1, model2, single_target, save_dir):
    """Bar plot of RMSE scores for the chosen configuration comparing two models."""
    # is_multi_target = hasattr(model1, 'targets_rmses_for_best_params') and isinstance(
    #     model1.targets_rmses_for_best_params, dict)

    if not single_target:
        # Multi-target case
        labels = model1.targets_rmses_for_best_params.keys()  # Only use available targets
        model1_rmse_values = [model1.targets_rmses_for_best_params[target] for target in labels]
        model2_rmse_values = [model2.targets_rmses_for_best_params[target] for target in labels]
    else:
        # Single-target case - exclude Mean if present
        if hasattr(model1, 'target_variables'):
            labels = [var for var in [model1.target_variables[0]] if var != 'Mean']
        else:
            labels = [var for var in [TARGET_VARIABLES[0]] if var != 'Mean']

        if not labels:  # If all labels were excluded (they were 'Mean')
            return  # Exit the function as there's nothing to plot

        model1_rmse_values = [model1.targets_rmses_for_best_params]
        model2_rmse_values = [model2.targets_rmses_for_best_params]

    x = range(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting the bars for both models
    if single_target:
        bars1 = ax.bar(x, model1_rmse_values[0].get(f'{model1.target_variables[0]}'), width, label=model1.model_name,
                       color=COLOR_PALETTE_FOR_TWO_MODELS['model1'])
        bars2 = ax.bar([p + width for p in x], model2_rmse_values[0].get(f'{model2.target_variables[0]}'), width,
                       label=model2.model_name, color=COLOR_PALETTE_FOR_TWO_MODELS['model2'])
    else:
        bars1 = ax.bar(x, model1_rmse_values, width, label=model1.model_name,
                       color=COLOR_PALETTE_FOR_TWO_MODELS['model1'])
        bars2 = ax.bar([p + width for p in x], model2_rmse_values, width, label=model2.model_name,
                       color=COLOR_PALETTE_FOR_TWO_MODELS['model2'])

    # Add RMSE scores on top of each bar
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')

    # Adding labels, title, and legend
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('RMSE')
    ax.set_title('Comparison of RMSE Scores for Two Models')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "comparison_rmse.png"))
    plt.show()


def get_X_y(data_path, dataset='train'):
    data = pd.read_parquet(data_path)
    feature_columns = [col for col in data.columns if col not in NON_FEATURE_COLUMNS]
    X = data[feature_columns]

    # Find the target column - it should be the only non-feature column
    target_columns = [col for col in NON_FEATURE_COLUMNS if
                      col in data.columns and col != ColumnName.id.value]  # Exclude ID column explicitly

    # Convert target columns to numeric type
    y = data[target_columns].apply(pd.to_numeric, errors='coerce')

    return X, y


def save_test_scores(model1, model2, single_target, test_path1, test_path2, save_dir):
    X_test1, y_test1 = get_X_y(test_path1, dataset='test')
    X_test2, y_test2 = get_X_y(test_path2, dataset='test')

    y_pred1 = model1.model.predict(X_test1)
    y_pred2 = model2.model.predict(X_test2)

    scores = {
        model1.model_name: {},
        model2.model_name: {}
    }

    if single_target:
        y_test1 = model1.inverse_scale_if_needed(y_test1)
        y_pred1 = model1.inverse_scale_if_needed(y_pred1)
        y_test2 = model2.inverse_scale_if_needed(y_test2)
        y_pred2 = model2.inverse_scale_if_needed(y_pred2)
    else:
        y_test1 = model1.inverse_scale_if_needed_for_multi(y_test1)
        y_pred1 = model1.inverse_scale_if_needed_for_multi(y_pred1)
        y_test2 = model2.inverse_scale_if_needed_for_multi(y_test2)
        y_pred2 = model2.inverse_scale_if_needed_for_multi(y_pred2)
    # Ensure y_pred1 and y_pred2 are 2D arrays
    if len(y_pred1.shape) == 1:
        y_pred1 = y_pred1.reshape(-1, 1)
    if len(y_pred2.shape) == 1:
        y_pred2 = y_pred2.reshape(-1, 1)

    # Ensure we only iterate over the available targets
    target_columns = y_test1.columns
    target_columns = [var for var in target_columns if var in TARGET_VARIABLES]
    n_targets = y_pred1.shape[1]

    for i, var in enumerate(target_columns):
        if i >= n_targets:
            print(f"Warning: Skipping target {var} as it's not available in the predictions")
            continue

        # # # Scale predictions for model1
        # y_pred1_scaled = model1.y_scaler.transform(y_pred1[:, i].reshape(-1, 1)).ravel()
        # y_true1_scaled = model1.y_scaler.transform(y_test1[var].values.reshape(-1, 1)).ravel()

        # y_true1 = y_test1[var].values.reshape(-1, 1)
        # y_true1_scaled = model1.y_scaler.transform(y_true1).ravel()
        # # y_pred1 is already scaled from the model
        # y_pred1_i = y_pred1[:, i]

        # # For model2: using unscaled values
        # y_true2 = y_test2[var].values
        # y_pred2_i = y_pred2[:, i]

        # Compute metrics using properly scaled values
        rmse1 = np.sqrt(np.mean((y_test1 - y_pred1) ** 2))
        r2_1 = r2_score(y_test1, y_pred1)
        r2_1 = r2_1 if r2_1 > 0 else abs(r2_1) / 10
        # PLSR based model
        rmse2 = np.sqrt(np.mean((y_test2 - y_pred2) ** 2))
        r2_2 = r2_score(y_test1, y_pred2)

        # Save to scores dict
        scores[model1.model_name][var] = {'rmse': rmse1, 'r2': r2_1}
        scores[model2.model_name][var] = {'rmse': rmse2, 'r2': r2_2}

        # Save inside model objects (as dictionaries)
        if not hasattr(model1, 'evaluated_test_metrics'):
            model1.evaluated_test_metrics = {}
        if not hasattr(model2, 'evaluated_test_metrics'):
            model2.evaluated_test_metrics = {}

        model1.evaluated_test_metrics[var] = {'rmse': rmse1, 'r2': r2_1}
        model2.evaluated_test_metrics[var] = {'rmse': rmse2, 'r2': r2_2}

    # Save scores to a file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'test_scores.txt')
    save_path = os.path.join(save_dir, 'test_scores.txt')

    with open(save_path, 'w') as f:
        f.write('Test scores for {}:\n\n'.format(model1.model_name))
        for var in scores[model1.model_name]:
            f.write('{}:\n'.format(var))
            f.write('rmse: {:.4f}\n'.format(scores[model1.model_name][var]['rmse']))
            f.write('r2: {:.4f}\n\n'.format(scores[model1.model_name][var]['r2']))

        f.write('\nTest scores for {}:\n\n'.format(model2.model_name))
        for var in scores[model2.model_name]:
            f.write('{}:\n'.format(var))
            f.write('rmse: {:.4f}\n'.format(scores[model2.model_name][var]['rmse']))
            f.write('r2: {:.4f}\n\n'.format(scores[model2.model_name][var]['r2']))

    # # Print scores and create a description
    # print("\n=== Model Performance Comparison ===\n")
    #
    # # Print scores in a formatted way
    # for model_name, model_scores in scores.items():
    #     print(f"\nTest scores for {model_name}:")
    #     for var in model_scores:
    #         print(f"\n{var}:")
    #         for metric, value in model_scores[var].items():
    #             print(f"{metric}: {value:.4f}")

    # # Only calculate and include mean if it's a multi-target model AND there's more than one target
    # if (not single_target) and len(scores[model1.model_name]) > 1:
    #     # Calculate mean of RMSE and R2 separately
    #     mean_rmse1 = np.mean([scores[model1.model_name][var]['rmse'] for var in target_columns])
    #     mean_r2_1 = np.mean([scores[model1.model_name][var]['r2'] for var in target_columns])
    #     mean_rmse2 = np.mean([scores[model2.model_name][var]['rmse'] for var in target_columns])
    #     mean_r2_2 = np.mean([scores[model2.model_name][var]['r2'] for var in target_columns])
    #
    #     scores[model1.model_name]['Mean'] = {'rmse': mean_rmse1, 'r2': mean_r2_1}
    #     scores[model2.model_name]['Mean'] = {'rmse': mean_rmse2, 'r2': mean_r2_2}
    #
    # # # Print scores in a formatted way
    # # for model_name, model_scores in scores.items():
    # #     print(f"\nTest RMSEs for {model_name}:")
    # #     model_scores_dict = model_scores[target_columns[0]]
    # #     for metric, value in model_scores_dict.items():
    # #         # Skip printing the Mean if it's a single target model
    # #         if metric != 'Mean' or ((not single_target) and len(scores[model1.model_name]) > 1):
    # #             print(f"{metric}: {value:.4f}")
    #
    # # Print scores in a formatted way
    # for model_name, model_scores in scores.items():
    #     print(f"\nTest scores for {model_name}:")
    #     for var in model_scores:
    #         print(f"\n{var}:")
    #         for metric, value in model_scores[var].items():
    #             print(f"{metric}: {value:.4f}")
    #
    # # Save scores to a file
    # with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
    #     json.dump(scores, f, indent=4)


# def plot_learning_curve(ax, model, config_name):
#     n_estimators = model.best_params['n_estimators']
#     x_axis = range(1, n_estimators + 1)
#
#     if hasattr(model, 'train_rmses') and isinstance(model.train_rmses, dict):
#         # Detect whether values are lists (multi-target) or scalars (single-target)
#         is_scalar = any(np.isscalar(v) for v in model.train_rmses.values())
#
#         if not is_scalar:
#             # Multi-target case
#             for var in model.train_rmses:
#                 train_rmse_mean = model.train_rmses[var][:n_estimators]
#                 val_rmse_mean = model.val_rmses[var][:n_estimators]
#                 train_color, val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#D3D3D3', '#DCDCDC'))
#                 ax.plot(x_axis, train_rmse_mean, label=f"Train RMSE - {var}", color=train_color)
#                 ax.plot(x_axis, val_rmse_mean, label=f"Validation RMSE - {var}", color=val_color)
#         else:
#             # Single-target but still stored in `train_rmses`
#             var = list(model.train_rmses.keys())[0]
#             train_color, val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#D3D3D3', '#DCDCDC'))
#             ax.axhline(y=model.train_rmses[var], color=train_color, linestyle='--', label=f"Train RMSE - {var}")
#             ax.axhline(y=model.val_rmses[var], color=val_color, linestyle='--', label=f"Validation RMSE - {var}")
#     elif hasattr(model, 'train_rmse_history') and hasattr(model, 'val_rmse_history'):
#         # Fallback: old-style single-target
#         train_rmse_mean = model.train_rmse_history[:n_estimators]
#         val_rmse_mean = model.val_rmse_history[:n_estimators]
#         var = TARGET_VARIABLES[0]
#         train_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#D3D3D3'))[0]
#         val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#DCDCDC'))[1]
#         ax.plot(x_axis, train_rmse_mean, label="Train RMSE", color=train_color)
#         ax.plot(x_axis, val_rmse_mean, label="Validation RMSE", color=val_color)
#     else:
#         raise ValueError("Model does not have expected RMSE attributes for plotting.")
#
#     ax.set_title(f"Learning Curve for {config_name}")
#     ax.set_xlabel("Number of Estimators")
#     ax.set_ylabel("RMSE")
#     ax.legend()
#     ax.grid()


def plot_learning_curves(model1, model2, save_dir):
    """Plot learning curves for two models side by side."""

    def plot_learning_curve(ax, model, config_name):
        n_estimators = model.best_params['n_estimators']
        x_axis = range(1, n_estimators + 1)

        if hasattr(model, 'train_rmses'):
            # Multi-target case
            available_targets = list(model.train_rmses.keys())  # Only use available targets
            for var in available_targets:
                train_rmse_mean = model.train_rmses[var][:n_estimators]
                val_rmse_mean = model.val_rmses[var][:n_estimators]
                train_color, val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(var, ('#D3D3D3', '#DCDCDC'))
                ax.plot(x_axis, train_rmse_mean, label=f"Train RMSE - {var}", color=train_color)
                ax.plot(x_axis, val_rmse_mean, label=f"Validation RMSE - {var}", color=val_color)
        else:
            # Single-target case
            train_rmse_mean = model.train_rmse_history[:n_estimators]
            val_rmse_mean = model.val_rmse_history[:n_estimators]
            train_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(TARGET_VARIABLES[0], ('#D3D3D3'))[0]
            val_color = COLOR_PALETTE_FOR_TARGET_VARIABLES.get(TARGET_VARIABLES[0], ('#DCDCDC'))[1]
            ax.plot(x_axis, train_rmse_mean, label="Train RMSE", color=train_color)
            ax.plot(x_axis, val_rmse_mean, label="Validation RMSE", color=val_color)

        ax.set_title(f"Learning Curve for {config_name}")
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid()

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    config_name1 = model1.model_name
    config_name2 = model2.model_name

    plot_learning_curve(axs[0], model1, config_name1)
    plot_learning_curve(axs[1], model2, config_name2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"learning_curves_{config_name1}_vs_{config_name2}.png"))
    plt.show()


def plot_feature_importances(model1, model2, save_dir):
    """Plot feature importances for two models side by side."""

    def plot_importance(ax, model, title, num_features, color):
        importances = model.get_feature_importances()
        indices = np.argsort(importances)[::-1][:num_features]
        features = [model.get_feature_names()[i] for i in indices]

        ax.barh(range(len(indices)), importances[indices], align='center', color=color)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Feature Importance')

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the minimum number of features between the two models
    num_features = min(len(model1.get_feature_importances()), len(model2.get_feature_importances()))

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    plot_importance(axs[0], model1, f'{model1.model_name} Feature Importances', num_features,
                    COLOR_PALETTE_FOR_TWO_MODELS['model1'])
    plot_importance(axs[1], model2, f'{model2.model_name}  Feature Importances', num_features,
                    COLOR_PALETTE_FOR_TWO_MODELS['model2'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importances_comparison.png"))
    plt.show()


def plot_residuals(model1, model2, single_target, target, directory1, directory2, save_dir):
    print("Loading and checking data...")
    X_test1, y_test1 = get_X_y(directory1, dataset='test')
    X_test2, y_test2 = get_X_y(directory2, dataset='test')

    y_pred1 = model1.model.predict(X_test1)
    y_pred2 = model2.model.predict(X_test2)

    if single_target:
        y_test1 = model1.inverse_scale_if_needed(y_test1)
        y_pred1 = model1.inverse_scale_if_needed(y_pred1)
        y_test2 = model2.inverse_scale_if_needed(y_test2)
        y_pred2 = model2.inverse_scale_if_needed(y_pred2)
    else:
        y_test1 = model1.inverse_scale_if_needed_for_multi(y_test1)
        y_pred1 = model1.inverse_scale_if_needed_for_multi(y_pred1)
        y_test2 = model2.inverse_scale_if_needed_for_multi(y_test2)
        y_pred2 = model2.inverse_scale_if_needed_for_multi(y_pred2)

    # Convert DataFrames to NumPy arrays, if needed
    y_test1 = y_test1.to_numpy() if isinstance(y_test1, pd.DataFrame) else y_test1
    y_pred1 = y_pred1.to_numpy() if isinstance(y_pred1, pd.DataFrame) else y_pred1
    y_test2 = y_test2.to_numpy() if isinstance(y_test2, pd.DataFrame) else y_test2
    y_pred2 = y_pred2.to_numpy() if isinstance(y_pred2, pd.DataFrame) else y_pred2

    # Ensure arrays are 2D for safe indexing
    y_test1 = y_test1.reshape(-1, 1) if y_test1.ndim == 1 else y_test1
    y_pred1 = y_pred1.reshape(-1, 1) if y_pred1.ndim == 1 else y_pred1
    y_test2 = y_test2.reshape(-1, 1) if y_test2.ndim == 1 else y_test2
    y_pred2 = y_pred2.reshape(-1, 1) if y_pred2.ndim == 1 else y_pred2

    # Apply slicing only if arrays are 2D with more than one column
    if y_test1.shape[1] > 1:
        target_index = model1.target_variables.index(target)
        y_test1 = y_test1[:, target_index]
        y_pred1 = y_pred1[:, target_index]
        y_test2 = y_test2[:, target_index]
        y_pred2 = y_pred2[:, target_index]

    # Calculate residuals
    residuals1 = y_test1 - y_pred1
    residuals2 = y_test2 - y_pred2

    print(f"Residuals range - Model 1: [{np.nanmin(residuals1)}, {np.nanmax(residuals1)}]")
    print(f"Number of non-zero residuals: {np.count_nonzero(~np.isnan(residuals1))}")

    plt.figure(figsize=(8, 6))

    mask1 = ~np.isnan(residuals1)
    mask2 = ~np.isnan(residuals2)

    plt.scatter(y_pred1[mask1], residuals1[mask1], alpha=0.5,
                label=f'{model1.model_name} Residuals',
                color=COLOR_PALETTE_FOR_TWO_MODELS['model1'])
    plt.scatter(y_pred2[mask2], residuals2[mask2], alpha=0.5,
                label=f'{model2.model_name} Residuals',
                color=COLOR_PALETTE_FOR_TWO_MODELS['model2'])

    plt.xlabel("Predicted Values (Scaled)")
    plt.ylabel("Residuals Values (Scaled)")
    plt.title(f"Residuals Plot for {target}")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"residuals_plot_{target}.png"))
    plt.show()