import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression


def bld_NIR_mdl(clb):
    # Identify columns with numeric names
    cs = [i for i, col in enumerate(clb.columns) if col.isdigit()]

    # Create training DataFrame
    trn = pd.DataFrame(clb.iloc[:, ~clb.columns.isin(cs)])
    trn['spc'] = clb.iloc[:, cs].values.tolist()

    ftnir_mdls = {}

    for mdl in trn['model'].unique():
        t1 = trn[trn['model'] == mdl]

        if t1['value'].notna().sum() < 10:
            continue

        # Apply Savitzky-Golay filter
        spc_filtered = np.apply_along_axis(lambda x: savgol_filter(x, 5, 2), 1, np.array(t1['spc'].tolist()))

        # Fit PLSR model
        pls = PLSRegression(n_components=10)
        pls.fit(spc_filtered, t1['value'])

        # Select optimal number of components
        nc = pls.n_components  # Placeholder for selectNcomp equivalent

        if nc < 1 or pls is None:
            continue

        print(f"{mdl} {nc}")

        pls.unit = t1['unit'].iloc[0]
        pls.variable = t1['variable'].iloc[0]
        pls.filter = 'nir'
        pls.nc = nc
        pls.model = f"{t1['unit'].iloc[0]} {t1['variable'].iloc[0]}"

        ftnir_mdls[mdl] = pls

    return ftnir_mdls
