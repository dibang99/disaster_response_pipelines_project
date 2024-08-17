import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge message and category CSV files into a single DataFrame.

    Args:
        messages_file (str): Path to the CSV file containing messages.
        categories_file (str): Path to the CSV file containing categories.

    Returns:
        pd.DataFrame: A DataFrame with merged messages and categories.
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages_df, categories_df, on='id')
    return merged_df


def clean_data(df):
    """Clean the DataFrame by transforming category data into separate columns and binary values.

    Args:
        df (pd.DataFrame): DataFrame with messages and category data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with individual category columns.
    """
    # Remove duplicate entries
    df = df.drop_duplicates()

    # Extract and expand categories into separate columns
    category_data = df['categories'].str.split(';', expand=True)
    category_names = [cat.split('-')[0] for cat in category_data.iloc[0]]
    category_data.columns = category_names

    # Convert category values to binary (0 or 1) and handle invalid values
    for column in category_data:
        category_data[column] = category_data[column].str[-1].astype(int)
        category_data = category_data[category_data[column] <= 1]

    # Drop the original categories column and append the new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, category_data], axis=1)

    # Display value counts for each category column
    for column in category_data:
        print(df[column].value_counts())
    
    return df


def save_data(df, database_filename):
    """Save the DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        database_path (str): The path to the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()