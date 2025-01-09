import pandas as pd
from constants_config import ColumnName, CropTissue, IDComponents


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    # Print the number of kinds of crops
    unique_crops = df[IDComponents.crop.value].unique()
    print(f"Number different crops: {len(unique_crops)}")
    print("With total number of rows: ", len(df))

    # Print the number of rows for each crop
    crop_counts = df[IDComponents.crop.value].value_counts()
    print("\nNumber of rows for each crop:")
    for crop, count in crop_counts.items():
        print(f"{IDComponents.crop.value}: {crop}, Rows: {count}")

    # Count the number of n_values, sc_value, and st_values for each crop and part
    crop_part_value_counts = df.groupby([IDComponents.crop.value, IDComponents.tissue.value],
                                        group_keys=False).apply(
        lambda x: pd.Series({
            ColumnName.n_value.value: x[[ColumnName.n_value.value, 'predicted_N_Value']].notna().any(axis=1).sum(),
            ColumnName.sc_value.value: x[[ColumnName.sc_value.value, 'predicted_SC_Value']].notna().any(axis=1).sum(),
            ColumnName.st_value.value: x[[ColumnName.st_value.value, 'predicted_ST_Value']].notna().any(axis=1).sum()
        })
    ).reset_index()

    for crop in df[IDComponents.crop.value].unique():
        crop_data = crop_part_value_counts[crop_part_value_counts[IDComponents.crop.value] == crop]
        parts = crop_data[IDComponents.tissue.value]
        n_values_counts = crop_data[ColumnName.n_value.value]
        sc_values_counts = crop_data[ColumnName.sc_value.value]
        st_values_counts = crop_data[ColumnName.st_value.value]

        # Print the parameters
        print(f"\n\n{IDComponents.crop.value}: {crop}")
        for tissue, n_val, sc_val, st_val in zip(parts, n_values_counts, sc_values_counts, st_values_counts):
            no_value_count = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.tissue.value] == tissue) &
                                df[[ColumnName.n_value.value, 'predicted_N_Value', ColumnName.sc_value.value,
                                    'predicted_SC_Value', ColumnName.st_value.value, 'predicted_ST_Value']].isna().all(
                                    axis=1)].shape[0]
            missing_n = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.tissue.value] == tissue) &
                           df[[ColumnName.n_value.value, 'predicted_N_Value']].isna().all(axis=1)].shape[0]
            missing_sc = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.tissue.value] == tissue) &
                            df[[ColumnName.sc_value.value, 'predicted_SC_Value']].isna().all(axis=1)].shape[0]
            missing_st = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.tissue.value] == tissue) &
                            df[[ColumnName.st_value.value, 'predicted_ST_Value']].isna().all(axis=1)].shape[0]

            if include_negatives:
                negative_counts = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.tissue.value] == tissue)][
                    [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].lt(0).sum()
                negative_counts = {k: v for k, v in negative_counts.items() if v > 0}
                if negative_counts:
                    print(
                        f"{IDComponents.tissue.value}: {tissue}, {ColumnName.n_value.value}: {n_val}, "
                        f"{ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}, Negative values: {negative_counts}")
                else:
                    print(
                        f"{IDComponents.tissue.value}: {tissue}, {ColumnName.n_value.value}: {n_val}, "
                        f"{ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}")
            else:
                print(
                    f"{IDComponents.tissue.value}: {tissue}, {ColumnName.n_value.value}: {n_val}, "
                    f"{ColumnName.sc_value.value}: {sc_val}, "
                    f"{ColumnName.st_value.value}: {st_val}")
            if no_value_count > 0:
                print(f"Rows with all values missing: {no_value_count}")
            if missing_n > 0:
                print(f"Rows with both {ColumnName.n_value.value} and predicted_N_Value missing: {missing_n}")
            if missing_sc > 0:
                print(f"Rows with both {ColumnName.sc_value.value} and predicted_SC_Value missing: {missing_sc}")
            if missing_st > 0:
                print(f"Rows with both {ColumnName.st_value.value} and predicted_ST_Value missing: {missing_st}")


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_rows_with_negatives(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_check = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]
    return df[(df[columns_to_check] >= 0).all(axis=1)]


def main():
    print("Analysis for all relevant crops:")
    df = pd.read_csv('data_files/extended_df.csv')
    analyze_crops(df, include_negatives=True)
    print("\nAnalysis is done for all crops.")


if __name__ == "__main__":
    main()
