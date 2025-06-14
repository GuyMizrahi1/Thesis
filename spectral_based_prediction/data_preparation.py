import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from constants_config import ColumnName, Crop, CropPart, IDComponents, DATA_FOLDER, TARGET_VARIABLES, target_variables, \
    current_crop


def filter_leaf_samples(df: pd.DataFrame) -> pd.DataFrame:
    # Extract crop and part from the ID column and convert to lowercase
    df.loc[:, IDComponents.crop.value] = df[ColumnName.id.value].str[:3].str.lower()
    df.loc[:, IDComponents.part.value] = df[ColumnName.id.value].str[6:9].str.lower()

    # Update incorrect classifier 'ea2' to 'lea'
    df.loc[df[IDComponents.part.value] == 'ea2', IDComponents.part.value] = CropPart.leaf.value

    # Filter the DataFrame to include only rows representing leaf samples
    df_leaf_samples = df[df[IDComponents.part.value] == CropPart.leaf.value]
    return df_leaf_samples


def analyze_explained_variables_per_crop(df: pd.DataFrame, include_negatives: bool):
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
            if include_negatives:
                negative_counts = df[(df[IDComponents.crop.value] == crop) & (df[IDComponents.part.value] == part)][
                    [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]].lt(0).sum()
                negative_counts = {k: v for k, v in negative_counts.items() if v > 0}
                if negative_counts:
                    print(
                        f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                        f"{ColumnName.st_value.value}: {st_val}, Negative values: {negative_counts}")
                else:
                    print(f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                          f"{ColumnName.st_value.value}: {st_val}")
            else:
                print(f"Part: {part}, {ColumnName.n_value.value}: {n_val}, {ColumnName.sc_value.value}: {sc_val}, "
                      f"{ColumnName.st_value.value}: {st_val}")


def analyze_crops(df: pd.DataFrame, include_negatives: bool = True):
    # Print number of unique crops
    unique_crops = df[ColumnName.crop.value].unique()
    print(f"Number of different crops: {len(unique_crops)}")
    print("Total number of rows:", len(df))

    # Get available target columns from the DataFrame
    target_columns = [col for col in [ColumnName.n_value.value,
                                      ColumnName.sc_value.value,
                                      ColumnName.st_value.value]
                      if col in df.columns]

    # Prepare a list to hold a row-wise summary
    summary_data = []

    for crop in unique_crops:
        crop_df = df[df[ColumnName.crop.value] == crop]

        # Initialize dictionary with crop name
        crop_summary = {'Crop': crop}

        # Count values for each available target column
        for col in target_columns:
            count_key = f'{col} Count'
            crop_summary[count_key] = crop_df[col].count()

            if include_negatives:
                neg_key = f'Negative {col.split("_")[0]}'  # Extract N, SC, or ST
                crop_summary[neg_key] = (crop_df[col] < 0).sum()

        # Calculate rows with no values (only for available columns)
        if target_columns:  # Only if there are target columns
            no_values = crop_df[target_columns].isna().all(axis=1).sum()
            crop_summary['Rows with No Values'] = no_values

        summary_data.append(crop_summary)

    # Convert to DataFrame for a nice tabular display
    summary_df = pd.DataFrame(summary_data)
    print("\nCrop Data Summary:\n")
    print(summary_df.to_string(index=False))


def remove_observations_with_missing_explained_variables(df: pd.DataFrame, explained_variables: List) -> pd.DataFrame:
    return df.dropna(subset=explained_variables)


def replace_negative_values_with_mean_or_median(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in ['mean', 'median']:
        raise ValueError("Method must be 'mean' or 'median'")

    # Get available target columns from the DataFrame
    target_columns = [col for col in [ColumnName.n_value.value,
                                      ColumnName.sc_value.value,
                                      ColumnName.st_value.value]
                      if col in df.columns]

    for column in target_columns:
        # Get only non-negative values for calculating replacement value
        non_negative_values = df[df[column] >= 0][column]

        # Skip if no non-negative values available
        if len(non_negative_values) == 0:
            print(f"Warning: Column '{column}' has no non-negative values to calculate {method}")
            continue

        # Calculate replacement value
        replacement_value = (non_negative_values.mean() if method == 'mean'
                             else non_negative_values.median())

        # Find and replace negative values
        negative_mask = df[column] < 0
        negative_count = negative_mask.sum()

        if negative_count > 0:
            example_negative_value = df.loc[negative_mask, column].iloc[0]
            df.loc[negative_mask, column] = replacement_value
            print(f"Column '{column}' had {negative_count} negative values. "
                  f"Method: {method}. Replacement value: {replacement_value:.4f}. "
                  f"Example of negative value: {example_negative_value:.4f}")

    return df


def split_dataset(df: pd.DataFrame, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15,
                  random_state: int = 42):
    if train_size + validation_size + test_size != 1.0:
        raise ValueError("The sum of train_size, validation_size, and test_size must be 1.0")

    # Split the data into train and temp sets
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)

    # Calculate the proportion of validation and test sizes relative to the temp set
    validation_proportion = validation_size / (validation_size + test_size)

    # Split the temp set into validation and test sets
    validation_df, test_df = train_test_split(temp_df, train_size=validation_proportion, random_state=random_state)

    return train_df, validation_df, test_df


def save_and_print_dataset_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame,
                                  folder_path: str = DATA_FOLDER):
    os.makedirs(folder_path, exist_ok=True)
    # Print the size of each file
    print(f"\nTrain set size: {train_df.shape[0]} rows")
    print(f"Validation set size: {validation_df.shape[0]} rows")
    print(f"Test set size: {test_df.shape[0]} rows")

    # Save the datasets to parquet files in the specified folder
    train_file_path = f'{folder_path}/train_data.parquet'
    validation_file_path = f'{folder_path}/validation_data.parquet'
    test_file_path = f'{folder_path}/test_data.parquet'

    train_df.to_parquet(train_file_path)
    validation_df.to_parquet(validation_file_path)
    test_df.to_parquet(test_file_path)


def main(crops: List[str], explained_variables: List[str]):
    # Read the merged_data.parquet file into a DataFrame
    csv_file_path = f'{DATA_FOLDER}/all_crops_include_vine_merged_without_negative.csv'
    df = pd.read_csv(csv_file_path)
    analyze_crops(df, include_negatives=True)

    print("\n\nRemoving observations from irrelevant crops")
    filtered_by_crop_df = df[df[ColumnName.crop.value].isin(crops)]
    analyze_crops(filtered_by_crop_df, include_negatives=True)

    # Drop irrelevant columns
    explained_variables_columns_to_remove = list(set(TARGET_VARIABLES) - set(explained_variables))
    updated_df = filtered_by_crop_df.drop(columns=explained_variables_columns_to_remove)
    analyze_crops(updated_df, include_negatives=True)

    print("\n\nRemoving observations with missing explained variables")
    updated_df = remove_observations_with_missing_explained_variables(updated_df, explained_variables)

    print("\nHandling observations with negative values")
    updated_df = replace_negative_values_with_mean_or_median(updated_df, method='median')
    analyze_crops(updated_df, include_negatives=True)

    final_df = updated_df.drop(columns=[ColumnName.crop.value, ColumnName.tissue.value])
    # Split the dataset into train, validation, and test sets
    train_df, validation_df, test_df = split_dataset(final_df)

    if len(explained_variables) > 1:
        folder_path = f'{DATA_FOLDER}/{crops[0]}_multiple_explained_variables'
    else:
        folder_path = f'{DATA_FOLDER}/{crops[0]}_{explained_variables[0]}'

    # Save the datasets to parquet files and print their sizes
    save_and_print_dataset_splits(train_df, validation_df, test_df, folder_path)


if __name__ == "__main__":
    main(crops=[current_crop], explained_variables=target_variables)
