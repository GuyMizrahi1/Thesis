import os
import copy
import pandas as pd
from spectral_based_prediction.constants_config import DATA_FOLDER_PATH, ColumnName


class Dataset:
    def __init__(self, train_file_name, validation_file_name, test_file_name, data_folder_spec, target_variables):
        self.data_folder_spec = data_folder_spec
        self.target_variables = target_variables
        self.ID_train, self.X_train, self.Y_train = self.splitDependentAndIndependent(train_file_name, data_folder_spec,
                                                                                      target_variables)
        self.ID_val, self.X_val, self.Y_val = self.splitDependentAndIndependent(validation_file_name, data_folder_spec,
                                                                                target_variables)
        self.ID_test, self.X_test, self.Y_test = self.splitDependentAndIndependent(test_file_name, data_folder_spec,
                                                                                   target_variables)

    def splitDependentAndIndependent(self, data_file_name, data_folder_spec, target_variables):
        data_folder = os.path.join(DATA_FOLDER_PATH, data_folder_spec)
        print(f'data_folder: {data_folder}')
        data = pd.read_parquet(os.path.join(data_folder, data_file_name))
        id = data['ID']
        X = data.drop(columns=[ColumnName.id.value] + target_variables)
        Y = data[target_variables]
        return id, X, Y

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)
