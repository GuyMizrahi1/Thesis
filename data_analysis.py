import pandas as pd
from constants_config import ColumnName, CropTissue, IDComponents


def filter_leaf_parts(df: pd.DataFrame) -> pd.DataFrame:
    # Extract crop and part from the ID column and convert to lowercase
    df[IDComponents.crop.value] = df[ColumnName.id.value].str[:3].str.lower()
    df[IDComponents.part.value] = df[ColumnName.id.value].str[6:9].str.lower()

    # Update incorrect classifier 'ea2' to 'lea'
    df.loc[df[IDComponents.part.value] == 'ea2', IDComponents.part.value] = CropTissue.leaf_short.value

    # Filter the DataFrame to include only rows representing leaf samples
    df_leaf_samples = df[df[IDComponents.part.value] == CropTissue.leaf_short.value]
    return df_leaf_samples


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    # Print number of unique crops
    unique_crops = df[ColumnName.crop.value].unique()
    print(f"Number of different crops: {len(unique_crops)}")
    print("Total number of rows:", len(df))

    # Prepare list to hold row-wise summary
    summary_data = []

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
            'Crop': crop,
            'N_Value Count': n_val,
            'SC_Value Count': sc_val,
            'ST_Value Count': st_val,
            'Negative N': negative_n,
            'Negative SC': negative_sc,
            'Negative ST': negative_st,
            'Rows with No Values': no_values
        })

    # Convert to DataFrame for nice tabular display
    summary_df = pd.DataFrame(summary_data)
    print("\nCrop Data Summary:\n")
    print(summary_df.to_string(index=False))


def main():
    print("Analysis for all relevant crops:")
    # df = pd.read_csv('data_files/extended_df_with_predicted_columns.csv')
    df = pd.read_csv('data_files/extended_df.csv')
    analyze_crops(df, include_negatives=True)
    print("\nAnalysis is done for all crops.")


if __name__ == "__main__":
    main()
