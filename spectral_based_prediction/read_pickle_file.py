import os
import joblib

base_dir = os.getcwd()
pickle_file_path = os.path.join(base_dir, 'PLSR/models/cit_SC_Value/pls_5_components.pkl')

model = joblib.load(pickle_file_path)
print(f'evaluation_metrics: {model.evaluation_metrics}')

# print(f'best_params: {model.best_params}')
#
# print(f'train_rmses: {model.train_rmses}')
# print(f'val_rmses: {model.val_rmses}')
#
# print(f'evaluated_val_rmses: {model.evaluated_val_rmses}')
# print(f'evaluated_test_rmses: {model.evaluated_test_rmses}')
