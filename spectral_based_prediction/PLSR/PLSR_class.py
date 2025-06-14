from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from spectral_based_prediction.baseline_for_training.baseModels import BaseModel


class PLSRModel(BaseModel):

    def __init__(self, dataset, param_grid=None, is_multi_output=False, target_variable_name=None):
        if is_multi_output:
            model = MultiOutputRegressor(PLSRegression())
        else:
            model = PLSRegression()
        super().__init__(dataset, model, param_grid, is_multi_output, target_variable_name)
