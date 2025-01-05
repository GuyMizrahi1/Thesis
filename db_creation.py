import re
import os
import sqlite3
import pandas as pd
from constants_config import ColumnName, CropTissue, Crop, TARGET_VARIABLES, NON_FEATURE_COLUMNS, ID_MAPPING
from data_analysis import filter_leaf_parts


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


def add_columns_from_id(df: pd.DataFrame) -> pd.DataFrame:
    # Extract components from the ID column and insert them at the desired positions
    temp_df = df.copy()
    df.insert(4, 'Crop', df[ColumnName.id.value].str[:3])
    df.insert(5, 'Location', df[ColumnName.id.value].str[3:6])

    # Separate rows with 'NA' in 'Location' and process them
    na_location_df = df[df['Location'].str.contains(r'^(NAl|UNl|BH|BHlֿ)', flags=re.IGNORECASE)]
    na_location_df['Location'] = na_location_df[ColumnName.id.value].str[3:5]
    na_location_df.insert(6, 'Tissue', na_location_df[ColumnName.id.value].str[5:8])
    na_location_df.insert(7, 'Date', na_location_df[ColumnName.id.value].str[8:16])
    na_location_df.insert(8, 'Sample', na_location_df[ColumnName.id.value].str[14:])

    # Process rows without 'NA' in 'Location'
    non_na_location_df = df[~df['Location'].str.contains(r'^(NAl|UNl|BH|BHlֿ)', flags=re.IGNORECASE)]
    non_na_location_df.insert(6, 'Tissue', non_na_location_df[ColumnName.id.value].str[6:9])
    non_na_location_df.insert(7, 'Date', non_na_location_df[ColumnName.id.value].str[9:17])
    non_na_location_df.insert(8, 'Sample', non_na_location_df[ColumnName.id.value].str[17:])

    # Combine the processed DataFrames
    df = pd.concat([na_location_df, non_na_location_df]).sort_index()

    # Map the extracted components using ID_MAPPING
    df['Crop'] = df['Crop'].map(ID_MAPPING).fillna(df['Crop'])
    df['Location'] = df['Location'].map(ID_MAPPING).fillna(df['Location'])
    df['Tissue'] = df['Tissue'].map(ID_MAPPING).fillna(df['Tissue'])

    short_date_df = df[df['Date'].str.contains('_')]
    short_date_df['Date'] = df['Date'].apply(lambda x: '20' + x[:6])
    short_date_df['Sample'] = df['Date'].str[6:] + df['Sample']
    df.update(short_date_df)

    long_tissue_df = df[df['Date'].str.contains('f')]
    long_tissue_df['Tissue'] = long_tissue_df['Tissue'] + long_tissue_df['Date'].str[0]
    long_tissue_df['Date'] = long_tissue_df['Date'].str[1:] + long_tissue_df['Sample'].str[0]
    long_tissue_df['Sample'] = long_tissue_df['Sample'].str[1:]
    df.update(long_tissue_df)

    df['Sample'] = df['Sample'].str.replace('_', '')
    # remove nulls from sample
    df = df[df['Sample'].notna() & (df['Sample'] != '')]

    # Create Non Timened Sample column and insert it at the desired position
    df.insert(9, 'Non_Timed_Sample', df['Crop'] + df['Location'] + df['Tissue'] + '_' + df['Sample'])

    return df


def main(read_from_dbs: bool = True):
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
    null_filler_df = pd.concat([first_null_filler_df, second_null_filler_df], ignore_index=True)

    # Iterate over the columns to be updated
    for col, filler_col in zip(['N_Value', 'SC_Value', 'ST_Value'], ['N', 'SC', 'ST']):
        # Update the null values in extended_df with values from null_filler_df based on matching 'ID'
        extended_df[col] = extended_df.apply(
            lambda row: null_filler_df.loc[null_filler_df['ID'] == row['ID'], filler_col].values[0]
            if pd.isna(row[col]) and not null_filler_df.loc[null_filler_df['ID'] == row['ID'], filler_col].isna().empty
            else row[col],
            axis=1
        )

    extended_df.to_csv('data_files/extended_df.csv', index=False)
    # Columns to check for None values
    columns_to_check = ['N_Value', 'SC_Value', 'ST_Value']

    # DataFrame with rows that have at least one None value in the specified columns
    df_with_nones = extended_df[extended_df[columns_to_check].isna().any(axis=1)]
    df_with_nones.to_csv('data_files/df_with_nones.csv', index=False)
    # DataFrame with rows that have no None values in the specified columns
    df_without_nones = extended_df[extended_df[columns_to_check].notna().all(axis=1)]

    # Sanity check to ensure there are no None values in the second DataFrame
    assert df_without_nones[columns_to_check].isna().sum().sum() == 0, "There are None values in df_without_nones"


if __name__ == "__main__":
    main()
