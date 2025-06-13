import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from spectral_based_prediction.baseline_for_training.Dataset import Dataset
from spectral_based_prediction.constants_config import DATA_FOLDER_PATH, data_folder_spec, target_variables, \
    n_components


# Convert to DataFrame
def create_df_structure(X, columns):
    return {columns[i]: X[:, i] for i in range(X.shape[1])}


if __name__ == '__main__':
    # Creating Dataset Instance
    train_file_name = 'train_data.parquet'
    validation_file_name = 'validation_data.parquet'
    test_file_name = 'test_data.parquet'

    dataset = Dataset(train_file_name, validation_file_name, test_file_name, data_folder_spec, target_variables)

    # Standardizing the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # joblib.dump(X_scaler, os.path.join('./models', data_folder_spec, 'X_scaler.pkl'))
    # joblib.dump(y_scaler, os.path.join('./models', data_folder_spec, 'y_scaler.pkl'))

    dataset.X_train[dataset.X_train.columns] = X_scaler.fit_transform(dataset.X_train.values)
    dataset.X_val[dataset.X_val.columns] = X_scaler.transform(dataset.X_val.values)
    dataset.X_test[dataset.X_test.columns] = X_scaler.transform(dataset.X_test.values)

    y_scaled = y_scaler.fit_transform(dataset.Y_train)

    # Fit PLSRegression directly (update n_components follows get_explained_variance.py output)
    pls = PLSRegression(n_components=n_components)
    pls.fit(dataset.X_train, y_scaled)

    model_name = f'pls_{n_components}_components.pkl'
    os.makedirs(os.path.join('./models', data_folder_spec), exist_ok=True)
    joblib.dump(pls, os.path.join('./models', data_folder_spec, model_name))
    joblib.dump(y_scaler, os.path.join('./models', data_folder_spec, 'y_scaler.pkl'))
    joblib.dump(X_scaler, os.path.join('./models', data_folder_spec, 'X_scaler.pkl'))

    # Transform data
    X_train_plsr = pls.transform(dataset.X_train.reset_index(drop=True))
    X_val_plsr = pls.transform(dataset.X_val.reset_index(drop=True))
    X_test_plsr = pls.transform(dataset.X_test.reset_index(drop=True))

    # Generate feature names manually
    plsr_columns = [f'PLS_Component_{i + 1}' for i in range(n_components)]

    X_train_plsr_df = pd.DataFrame(create_df_structure(X_train_plsr, plsr_columns))
    X_val_plsr_df = pd.DataFrame(create_df_structure(X_val_plsr, plsr_columns))
    X_test_plsr_df = pd.DataFrame(create_df_structure(X_test_plsr, plsr_columns))

    # Combine with Y data
    train_data_plsr = pd.concat([pd.DataFrame(y_scaled, columns=dataset.Y_train.columns), X_train_plsr_df], axis=1)
    val_data_plsr = pd.concat(
        [pd.DataFrame(y_scaler.transform(dataset.Y_val), columns=dataset.Y_val.columns), X_val_plsr_df], axis=1)
    test_data_plsr = pd.concat(
        [pd.DataFrame(y_scaler.transform(dataset.Y_test), columns=dataset.Y_test.columns), X_test_plsr_df], axis=1)

    # Add ID column and set index
    train_data_plsr['ID'] = dataset.ID_train.reset_index(drop=True)
    val_data_plsr['ID'] = dataset.ID_val.reset_index(drop=True)
    test_data_plsr['ID'] = dataset.ID_test.reset_index(drop=True)

    train_data_plsr.set_index('ID', inplace=True)
    val_data_plsr.set_index('ID', inplace=True)
    test_data_plsr.set_index('ID', inplace=True)

    # Save transformed dataset
    output_data_folder = os.path.join(DATA_FOLDER_PATH, data_folder_spec)
    train_data_plsr.reset_index().to_parquet(
        os.path.join(output_data_folder, f'train_data_{n_components}_plsr.parquet'))
    val_data_plsr.reset_index().to_parquet(
        os.path.join(output_data_folder, f'validation_data_{n_components}_plsr.parquet'))
    test_data_plsr.reset_index().to_parquet(os.path.join(output_data_folder, f'test_data_{n_components}_plsr.parquet'))
