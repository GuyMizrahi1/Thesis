import re
import os
import sqlite3
import pandas as pd
from constants_config import (ColumnName, CropTissue, Crop, TARGET_VARIABLES, NON_FEATURE_COLUMNS, ID_MAPPING,
                              IDComponents)


def organize_result_data(df: pd.DataFrame) -> pd.DataFrame:
    df_without_nulls = df[~df['time'].isna()]
    pivot_df = df_without_nulls.pivot_table(index=[ColumnName.id.value], columns='variable', values='value',
                                            aggfunc='first')
    pivot_df.columns = [f"{col}_Value" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    return pivot_df


def organize_scan_data(df: pd.DataFrame) -> pd.DataFrame:
    df_headers = pd.concat([pd.Series([ColumnName.id.value]), df[df['filter'] == 'WN'].iloc[0, 2:]])
    observations_df = df[df['filter'] == 'A'].drop(columns=['filter'])
    observations_df.columns = df_headers
    return observations_df


def read_table(db_path, table_name) -> pd.DataFrame:
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f'PRAGMA table_info("{table_name}");')
    columns = [info[1] for info in cursor.fetchall()]

    cursor.execute(f'SELECT * FROM "{table_name}";')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No rows found in table {table_name}.")
        return pd.DataFrame()
    else:
        df = pd.DataFrame(rows, columns=columns)
        return df


def adjust_avo_df(avo_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    # Edit the ID column to add 'lea' after the 6th letter
    avo_df[ColumnName.id.value] = (avo_df[ColumnName.id.value].str[:6] + CropTissue.leaf.value +
                                   avo_df[ColumnName.id.value].str[6:])

    # Rename the 'N_cont' column to 'N_Value'
    avo_df.rename(columns={'N_cont': ColumnName.n_value.value}, inplace=True)

    # Adjust column names from 1 to 1557 by the column names of the floats in the merged_df
    float_columns = merged_df.select_dtypes(include=['float']).columns[3:1561]
    avo_df.columns = list(avo_df.columns[:2]) + list(float_columns)

    # Align columns with merged_df
    avo_df = avo_df.reindex(columns=merged_df.columns, fill_value=None)

    return avo_df


def adjust_alm_df(alm_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    # Create the ID column
    alm_df['Tissue sampling date/year'] = pd.to_datetime(alm_df['Tissue sampling date/year'], format='%d/%m/%Y',
                                                         errors='coerce')
    alm_df['id'] = alm_df['id'].fillna(alm_df['Original sample #'])
    alm_df[ColumnName.id.value] = (
            alm_df['Class_PLS_DA'].str[:3] +
            alm_df['Growing location'].str[:3] +
            alm_df['Examined tissue'].str[:3] +
            alm_df['Tissue sampling date/year'].dt.strftime('%Y%m%d') +
            '_' +
            alm_df['id'].astype(str)
    )

    # Remove rows with null IDs
    alm_df = alm_df.dropna(subset=[ColumnName.id.value])

    # Remove spaces from the ID column
    alm_df[ColumnName.id.value] = alm_df[ColumnName.id.value].str.replace(' ', '')

    # Handle duplicate IDs
    duplicate_ids = alm_df[ColumnName.id.value].duplicated(keep=False)
    duplicates = alm_df[duplicate_ids]
    for idx, row in duplicates.iterrows():
        original_id = row[ColumnName.id.value]
        technician = row['Technician']
        suffix = ''
        if technician in ['Ofek Woldenberg', 'Yeroslav']:
            location_within_tree = row['Location within the tree']
            if pd.notna(location_within_tree) and len(location_within_tree) >= 3:
                suffix = location_within_tree[:3]
            else:
                suffix = str(idx)
        elif pd.notna(row['Instrument sample #']) and len(row['Instrument sample #']) >= 5:
            suffix = row['Instrument sample #'][3:5]

        new_id = original_id + suffix
        while new_id in alm_df[ColumnName.id.value].values:
            suffix += '1'
            new_id = original_id + suffix

        alm_df.at[idx, ColumnName.id.value] = new_id

    id_column = alm_df.pop(ColumnName.id.value)
    alm_df.insert(0, ColumnName.id.value, id_column)

    # Rename the columns
    alm_df.rename(columns={
        'Analytical value (N)': ColumnName.n_value.value,
        'Analytical value (SC)': ColumnName.sc_value.value,
        'Analytical value (St)': ColumnName.st_value.value
    }, inplace=True)

    # Select the required columns
    numerical_columns = alm_df.columns[alm_df.columns.get_loc('3999.64'):]
    selected_columns = NON_FEATURE_COLUMNS + list(numerical_columns)
    alm_df = alm_df[selected_columns]

    # Remove rows where all three values of N_Value, SC_Value, and ST_Value are null
    alm_df = alm_df.dropna(subset=TARGET_VARIABLES, how='all')

    # Update column names of alm_df to match merged_df without altering values
    alm_df.columns = merged_df.columns

    return alm_df


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_rows_with_negatives(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_check = TARGET_VARIABLES
    return df[(df[columns_to_check] >= 0).all(axis=1)]


def read_data_from_dbs(read_from_dbs) -> pd.DataFrame:
    parquet_file_path = 'data_files/merged_data.parquet'
    if read_from_dbs:
        # Read and organize data from the first database
        result_db_path = 'DBS/RSL.db'
        ftnir_table_name = 'FTNIR'
        result_df = read_table(result_db_path, ftnir_table_name)
        organized_result_df = organize_result_data(result_df)

        # Read and organize data from the second database
        scan_db_path = 'DBS/SCN.db'
        ftnir_short_table_name = 'FTNIR_SHORT'
        scan_df = read_table(scan_db_path, ftnir_short_table_name)
        organized_scan_df = organize_scan_data(scan_df)

        # Merge the two DataFrames
        merged_df = organized_result_df.merge(organized_scan_df, on=ColumnName.id.value, how='outer')
        merged_df.to_parquet(parquet_file_path)
    else:
        merged_df = pd.read_parquet(parquet_file_path)
    return merged_df


def add_avo_alm_data(df: pd.DataFrame) -> pd.DataFrame:
    avo_data_path = 'data_files/avocado_FTNIR_tarin_lab.xlsx'
    alm_data_path = 'data_files/almond_FTNIR_tarin_lab.csv'
    avo_df = pd.read_excel(avo_data_path)
    alm_df = pd.read_csv(alm_data_path)

    # Adjust the data to fit the rest of the data structure
    adjusted_avo_df = adjust_avo_df(avo_df, df)
    adjusted_alm_df = adjust_alm_df(alm_df, df)

    # Merge the new data with the existing merged_df
    merged_df = pd.concat([df, adjusted_avo_df, adjusted_alm_df], ignore_index=True)

    return merged_df


def filter_crops(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[ColumnName.id.value].str[:3].isin([Crop.almond_short.value, Crop.avocado_short.value,
                                                  Crop.citrus_short.value, Crop.vine_short.value])]
    return df


def filter_leaf_parts(df: pd.DataFrame) -> pd.DataFrame:
    # Extract crop and part from the ID column and convert to lowercase
    df[IDComponents.crop.value] = df[ColumnName.id.value].str[:3].str.lower()
    df[IDComponents.tissue.value] = df[ColumnName.id.value].str[6:9].str.lower()

    # Update incorrect classifier 'ea2' to 'lea'
    df.loc[df[IDComponents.tissue.value] == 'ea2', IDComponents.tissue.value] = CropTissue.leaf_short.value

    # Filter the DataFrame to include only rows representing leaf samples
    df_leaf_samples = df[df[IDComponents.tissue.value] == CropTissue.leaf_short.value]
    return df_leaf_samples


def extract_id_components(df: pd.DataFrame) -> pd.DataFrame:
    df[IDComponents.crop.value] = df[ColumnName.id.value].str[:3]
    df[IDComponents.location.value] = df[ColumnName.id.value].str[3:6]
    return df


def process_na_location_rows(df: pd.DataFrame) -> pd.DataFrame:
    na_location_df = df[df[IDComponents.location.value].str.contains(r'^(NAl|UNl|BH|BHlֿ)', flags=re.IGNORECASE)]
    na_location_df[IDComponents.location.value] = na_location_df[ColumnName.id.value].str[3:5]
    na_location_df[IDComponents.tissue.value] = na_location_df[ColumnName.id.value].str[5:8]
    na_location_df[IDComponents.date.value] = na_location_df[ColumnName.id.value].str[8:16]
    na_location_df[IDComponents.sample.value] = na_location_df[ColumnName.id.value].str[14:]
    return na_location_df


def process_non_na_location_rows(df: pd.DataFrame) -> pd.DataFrame:
    non_na_location_df = df[~df[IDComponents.location.value].str.contains(r'^(NAl|UNl|BH|BHlֿ)', flags=re.IGNORECASE)]
    non_na_location_df[IDComponents.tissue.value] = non_na_location_df[ColumnName.id.value].str[6:9]
    non_na_location_df[IDComponents.date.value] = non_na_location_df[ColumnName.id.value].str[9:17]
    non_na_location_df[IDComponents.sample.value] = non_na_location_df[ColumnName.id.value].str[17:]
    return non_na_location_df


def map_id_components(df: pd.DataFrame) -> pd.DataFrame:
    df[IDComponents.crop.value] = df[IDComponents.crop.value].map(ID_MAPPING).fillna(df[IDComponents.crop.value])
    df[IDComponents.location.value] = df[IDComponents.location.value].map(ID_MAPPING).fillna(
        df[IDComponents.location.value])
    df[IDComponents.tissue.value] = df[IDComponents.tissue.value].map(ID_MAPPING).fillna(df[IDComponents.tissue.value])
    return df


def handle_special_cases(df: pd.DataFrame) -> pd.DataFrame:
    short_date_df = df[df[IDComponents.date.value].str.contains('_')]
    short_date_df[IDComponents.date.value] = short_date_df[IDComponents.date.value].apply(lambda x: '20' + x[:6])
    short_date_df[IDComponents.sample.value] = short_date_df[IDComponents.date.value].str[6:] + short_date_df[
        IDComponents.sample.value]
    df.update(short_date_df)

    long_tissue_df = df[df[IDComponents.date.value].str.contains('f')]
    long_tissue_df[IDComponents.tissue.value] = long_tissue_df[IDComponents.tissue.value] + long_tissue_df[
        IDComponents.date.value].str[0]
    long_tissue_df[IDComponents.date.value] = long_tissue_df[IDComponents.date.value].str[1:] + long_tissue_df[
        IDComponents.sample.value].str[0]
    long_tissue_df[IDComponents.sample.value] = long_tissue_df[IDComponents.sample.value].str[1:]
    df.update(long_tissue_df)

    return df


def add_non_timed_sample_column(df: pd.DataFrame) -> pd.DataFrame:
    df[IDComponents.sample.value] = df[IDComponents.sample.value].str.replace('_', '')
    df = df[df[IDComponents.sample.value].notna() & (df[IDComponents.sample.value] != '')]
    non_timed_sample_column = df[IDComponents.crop.value] + df[IDComponents.tissue.value] + df[
        IDComponents.location.value] + '_' + df[IDComponents.sample.value]
    df.insert(9, 'Non_Timed_Sample', non_timed_sample_column)
    return df


def get_row_with_most_values(group: pd.DataFrame, value_columns: list) -> pd.Series:
    # Count non-null values in the specified columns
    non_null_counts = group[value_columns].notna().sum(axis=1)
    # Get the index of the row with the maximum count of non-null values
    max_index = non_null_counts.idxmax()
    return group.loc[max_index]


def remove_duplicates_with_most_values(df: pd.DataFrame, id_column: str, value_columns: list) -> pd.DataFrame:
    # Group by the ID column and apply the function to each group
    deduplicated_df = df.groupby(id_column).apply(
        lambda group: get_row_with_most_values(group, value_columns)).reset_index(drop=True)
    return deduplicated_df


def main(read_from_dbs: bool = True, separate_filler_to_different_columns = False) -> None:

    # Read data from the databases or from the parquet file
    based_on_db_df = read_data_from_dbs(read_from_dbs)

    # Add avocado and almond data to the merged DataFrame
    merged_with_avo_and_alm_df = add_avo_alm_data(based_on_db_df)

    # Filter the crops to include only almond, avocado, citrus, and vine
    filtered_crops_df = filter_crops(merged_with_avo_and_alm_df)

    # Preprocess the data to include only leaf samples
    leaf_samples_df = filter_leaf_parts(filtered_crops_df)

    # Add columns that inferred from the ID column
    extended_df = add_columns_from_id(leaf_samples_df)

    # filled missing values based on Or & Aviad's application
    first_null_filler_df = pd.read_csv('data_files/first_filler.csv')
    second_null_filler_df = pd.read_csv('data_files/second_filler.csv')
    vine_null_filler_df = pd.read_csv('data_files/vine_filler.csv')
    null_filler_df = pd.concat([first_null_filler_df, second_null_filler_df, vine_null_filler_df], ignore_index=True)
    null_filler_df = remove_duplicates_with_most_values(null_filler_df, 'ID', ['N', 'SC', 'ST'])

    if separate_filler_to_different_columns:
        # Mapping from base values in extended_df to model predictions from null_filler_df
        value_to_prediction = {'N_Value': 'N_ESTIMATED', 'SC_Value': 'SC_ESTIMATED', 'ST_Value': 'ST_ESTIMATED'}

        # Source columns in null_filler_df
        source_columns = {'N_Value': 'N', 'SC_Value': 'SC', 'ST_Value': 'ST'}

        # Start inserting at column index 4
        insert_pos = 4

        for original_col, new_col in value_to_prediction.items():
            source_col = source_columns[original_col]
            new_series = extended_df.apply(
                lambda row: null_filler_df.loc[null_filler_df['ID'] == row['ID'], source_col].values[0]
                if pd.isna(row[original_col]) and not null_filler_df.loc[
                    null_filler_df['ID'] == row['ID'], source_col].isna().all()
                # else row[original_col],
                else None,
                axis=1
            )
            extended_df.insert(loc=insert_pos, column=new_col, value=new_series)
            insert_pos += 1

        extended_df.drop(columns=['crop', 'part'], inplace=True)
        extended_df.to_csv('data_files/extended_df_with_predicted_columns.csv', index=False)
    else:
        # Iterate over the columns to be updated
        for col, filler_col in zip(['N_Value', 'SC_Value', 'ST_Value'], ['N', 'SC', 'ST']):
            # Update the null values in extended_df with values from null_filler_df based on matching 'ID'
            extended_df[col] = extended_df.apply(
                lambda row: null_filler_df.loc[null_filler_df['ID'] == row['ID'], filler_col].values[0]
                if pd.isna(row[col]) and not null_filler_df.loc[null_filler_df['ID'] == row['ID'], filler_col].isna().empty
                else row[col],
                axis=1
            )
        extended_df.drop(columns=['crop', 'part'], inplace=True)
        extended_df.to_csv('data_files/extended_df.csv', index=False)

    # Columns to check for None values
    columns_to_check = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]

    # DataFrame with rows that have at least one None value in the specified columns
    df_with_nones = extended_df[extended_df[columns_to_check].isna().any(axis=1)]
    # Keep the relevant rows - Avocado crop
    df_with_nones = df_with_nones[df_with_nones['Crop'] == Crop.avocado.value]
    df_with_nones.to_csv('data_files/avocado_df_with_nones_kabri_and_gilat.csv', index=False)

    # DataFrame with rows that have no None values in the specified columns
    df_without_nones = extended_df[extended_df[columns_to_check].notna().all(axis=1)]

    # Sanity check to ensure there are no None values in the second DataFrame
    assert df_without_nones[columns_to_check].isna().sum().sum() == 0, "There are None values in df_without_nones"
    print("Number of rows with at least one None value in the specified columns:", df_with_nones.shape[0])
    print("Number of rows with no None values in the specified columns:", df_without_nones.shape[0])

    extended_df.to_csv('data_files/extended_df.csv', index=False)


if __name__ == "__main__":
    main()
