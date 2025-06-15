import pandas as pd
from constants_config import ColumnName, CropTissue, IDComponents


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    # Print number of unique crops
    unique_crops = df[ColumnName.crop.value].unique()
    print(f"\nNumber of different crops: {len(unique_crops)}")
    print("Total number of rows:", len(df))

    # Prepare a list to hold a row-wise summary
    summary_data = []
    crop_full_name = {'alm': 'Almond', 'avo': 'Avocado', 'cit': 'Citrus', 'vin': 'Vine'}
    for crop in unique_crops:
        crop_df = df[df[ColumnName.crop.value] == crop]

        n_val = crop_df[ColumnName.n_value.value].count()
        sc_val = crop_df[ColumnName.sc_value.value].count()
        st_val = crop_df[ColumnName.st_value.value].count()

        no_values = crop_df[
            crop_df[[ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].isna().all(axis=1)
        ].shape[0]

        if include_negatives:
            negatives = crop_df[
                [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]
            ].lt(0).sum()
            negative_n = negatives[ColumnName.n_value.value]
            negative_sc = negatives[ColumnName.sc_value.value]
            negative_st = negatives[ColumnName.st_value.value]
        else:
            negative_n = negative_sc = negative_st = 0
        summary_data.append({
            'Crop': crop_full_name[crop],
            'N_Value Count': n_val,
            'SC_Value Count': sc_val,
            'ST_Value Count': st_val,
            'Negative N': negative_n,
            'Negative SC': negative_sc,
            'Negative ST': negative_st,
            'Rows with No Values': no_values
        })

    # Convert to DataFrame for a nice tabular display
    summary_df = pd.DataFrame(summary_data)
    print("\nCrop Data Summary:\n")
    print(summary_df.to_string(index=False))


def main():
    print("Analysis for all relevant crops:")
    # df = pd.read_csv('data_files/extended_df_with_predicted_columns.csv')
    df = pd.read_csv('data_files/all_crops_merged.csv')
    analyze_crops(df, include_negatives=True)
    print("\nAnalysis is done for all crops.")
    print('Remove rows without all three values...')
    valid_df = df[
        df[[ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].notna().any(axis=1)]
    valid_df.to_csv('data_files/valid_df.csv', index=False)
    analyze_crops(valid_df, include_negatives=True)

    # remove Negative values where
    final_df = valid_df[(valid_df[ColumnName.n_value.value] >= 0) &
                        ((valid_df[ColumnName.sc_value.value] >= 0) | (valid_df[ColumnName.sc_value.value].isna())) &
                        ((valid_df[ColumnName.st_value.value] >= 0) | (valid_df[ColumnName.st_value.value].isna()))]
    final_df.to_csv('data_files/final_df.csv', index=False)
    analyze_crops(final_df, include_negatives=True)
    print("Analysis is done for all crops.")


if __name__ == "__main__":
    main()
