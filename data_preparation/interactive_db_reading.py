import os
import zipfile
import sqlite3
import pandas as pd


def organize_data(df: pd.DataFrame) -> pd.DataFrame:
    df_without_nulls = df[~df['time'].isna()]

    pivot_df = df_without_nulls.pivot_table(index=['ID'], columns='variable', values='value', aggfunc='first')

    pivot_df.columns = [f"{col}_Value" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    return pivot_df


def unzip_files(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {zip_path} to {extract_to}")


def read_db_file(db_path):
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    if not tables:
        print(f"No tables found in {os.path.basename(db_path)}.")
        return []
    else:
        table_names = [table[0] for table in tables]
        print(f"Tables in {os.path.basename(db_path)}: {table_names}")
        return table_names


def read_table(db_path, table_name) -> pd.DataFrame:
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enclose table name in double quotes to handle spaces and special characters
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


def interactive_unzip():
    while True:
        unzip_more = input("Would you like to unzip a file? (yes/no): ").strip().lower()
        if unzip_more == 'yes':
            zip_path = input("Enter the path to the zip file: ").strip()
            extract_to = input("Enter the extraction directory: ").strip()
            unzip_files(zip_path, extract_to)
        else:
            break


def interactive_read_table(db_path, tables):
    while True:
        table_name = input(f"Which table would you like to read? {tables} (or type 'exit' to finish): ").strip()
        if table_name.lower() == 'exit':
            break
        if table_name in tables:
            df = read_table(db_path, table_name)
            return df
        else:
            print(f"Table {table_name} does not exist in the database. Please try again.")
            return pd.DataFrame()


def main():
    interactive_unzip()

    db_path = input("Enter the path to the database file: ").strip()
    tables = read_db_file(db_path)
    if tables:
        interactive_read_table(db_path, tables)


if __name__ == "__main__":
    main()
