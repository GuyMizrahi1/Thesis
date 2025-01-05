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
    # Print the number of kinds of crops
    unique_crops = df[IDComponents.crop.value].unique()
    print(f"Number different crops: {len(unique_crops)}")
    print("With total number of rows: ", len(df))

    # Print the number of rows for each crop
    crop_counts = df[IDComponents.crop.value].value_counts()
    print("\nNumber of rows for each crop:")
    for crop, count in crop_counts.items():
        print(f"Crop: {crop}, Rows: {count}")

    # Count the number of n_values, sc_value, and st_values for each crop and part
    crop_part_value_counts = df.groupby([IDComponents.crop.value, IDComponents.part.value]).agg({
        ColumnName.n_value.value: 'count',
        ColumnName.sc_value.value: 'count',
        ColumnName.st_value.value: 'count'
    }).reset_index()

    for crop in df[IDComponents.crop.value].unique():
        crop_data = crop_part_value_counts[crop_part_value_counts[IDComponents.crop.value] == crop]
        parts = crop_data[IDComponents.part.value]
        n_values_counts = crop_data[ColumnName.n_value.value]
        sc_values_counts = crop_data[ColumnName.sc_value.value]
        st_values_counts = crop_data[ColumnName.st_value.value]

        # Print the parameters
        print(f"\nCrop: {crop}")
        for part, n_val, sc_val, st_val in zip(parts, n_values_counts, sc_values_counts, st_values_counts):
            no_value_count = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.part.value] == part) &
                                df[[ColumnName.n_value.value, ColumnName.sc_value.value,
                                    ColumnName.st_value.value]].isna().all(axis=1)].shape[0]
            if include_negatives:
                negative_counts = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.part.value] == part)][
                    [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].lt(0).sum()
                negative_counts = {k: v for k, v in negative_counts.items() if v > 0}
                if negative_counts:
                    print(
                        f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}, Negative values: {negative_counts}, "
                        f" Amount of raws with no values: {no_value_count}")
                else:
                    print(
                        f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}, Amount of raws with no values: {no_value_count}")
            else:
                print(
                    f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                    f"{ColumnName.st_value.value}: {st_val}, Amount of raws with no values: {no_value_count}")


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
