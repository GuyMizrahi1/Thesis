import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from spectral_based_prediction.baseline_for_training.Dataset import Dataset
from spectral_based_prediction.constants_config import data_folder_spec, target_variables


def find_elbow(curve):
    """
    Finds the elbow point using the maximum curvature method.
    """
    # Get coordinates
    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T

    # Vector from first to last point
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    # Vector from point to first point
    vec_from_first = all_coord - first_point

    # Distance from point to line
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))

    # Find elbow point (maximum distance from line)
    idx_of_best_point = np.argmax(dist_to_line)
    return idx_of_best_point + 1  # Adding 1 because we start counting components from 1


if __name__ == '__main__':
    # Load the non-scaled dataset
    train_file_name = 'train_data.parquet'
    validation_file_name = 'validation_data.parquet'
    test_file_name = 'test_data.parquet'

    dataset = Dataset(train_file_name, validation_file_name, test_file_name, data_folder_spec, target_variables)

    # Create StandardScaler for X only, since we only need it for training and predictions
    X_scaler = StandardScaler()

    # Fit X scaler on training data and transform features
    X_scaled = X_scaler.fit_transform(dataset.X_train)

    # Number of components to test
    max_components = 50
    explained_variance_per_target = np.zeros((max_components, dataset.Y_train.shape[1]))

    # Compute explained variance for different numbers of components
    for n in tqdm(range(1, max_components + 1)):
        pls = PLSRegression(n_components=n)
        multi_pls = MultiOutputRegressor(pls)
        multi_pls.fit(X_scaled, dataset.Y_train)  # Use the raw Y_train directly

        # Predict on scaled X_train
        Y_pred = multi_pls.predict(X_scaled)

        # Compute explained variance (R²) directly on raw Y values (since they’re unscaled)
        explained_variance_per_target[n - 1, :] = [
            np.corrcoef(Y_pred[:, i], dataset.Y_train.iloc[:, i])[0, 1] ** 2
            for i in range(dataset.Y_train.shape[1])
        ]

    # Plot elbow curves for each target variable
    plt.figure(figsize=(10, 6))
    max_elbow_point=0
    for i in range(dataset.Y_train.shape[1]):
        variance_curve = explained_variance_per_target[:, i]
        plt.plot(range(1, max_components + 1), variance_curve, label=f'{target_variables[i]}')

        # Find and plot the elbow point
        elbow_point = find_elbow(variance_curve)
        plt.plot(elbow_point, variance_curve[elbow_point - 1], 'ro', label=f'Elbow Point (n={elbow_point})')
        plt.axvline(x=elbow_point, color='r', linestyle='--', alpha=0.3)

        print(f"Optimal number of components for {target_variables[i]}: {elbow_point}")
        if len(target_variables) > 1:
            if max_elbow_point < elbow_point:
                max_elbow_point = elbow_point
    print(f"\nNumber of components for Multioutput Model (Maximal Elbow Point): {max_elbow_point}")


    plt.xlabel('Number of PLS Components')
    plt.ylabel('Explained Variance (R²)')
    plt.title('Elbow Plot of Explained Variance in Target Variables')
    plt.legend(loc='lower right')
    plt.grid(True)
    os.makedirs(f'outputs/{data_folder_spec}', exist_ok=True)
    plt.savefig(f'outputs/{data_folder_spec}/explained_variance.png')