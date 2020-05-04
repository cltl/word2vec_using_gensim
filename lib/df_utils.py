import pandas as pd

def load_data(csv_path,
              remove_missing_values=True,
              verbose=0):
    df = pd.read_csv(csv_path)

    if verbose >= 2:
        print()
        print(f'loaded {csv_path}')
        print(df.shape)
        print(df.head())

    if remove_missing_values:
        df = df.dropna().reset_index(drop=True)
        for num_na_of_column in df.isnull().sum():
            assert num_na_of_column == 0, f'there appear to be missing values. Please inspect'

    return df

def load_cleaned_df(cleaned_txt, verbose=0):
    df_clean = pd.DataFrame({'clean': cleaned_txt})
    df_clean = df_clean.dropna().drop_duplicates()

    if verbose >= 2:
        print()
        print('txt df has the following shape')
        print(df_clean.shape)

    return df_clean
