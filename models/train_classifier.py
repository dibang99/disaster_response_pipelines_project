import sys
import pandas as pd

import nltk
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import pickle

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """Load message and category data from an SQLite database.

    Args:
        database_path (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Series with message text.
            - pd.DataFrame: DataFrame with category columns.
            - list: List of category names.
    """
    # Create a database engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the data from the SQLite table into a DataFrame
    df = pd.read_sql_table('messages', engine)

    # Optional: Limit the data for testing purposes (remove in production)
    df = df.head(100)
    
    # Extract messages and category data
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    # Get the list of category names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """Tokenize and lemmatize text.

    Args:
        text (str): Text to tokenize and lemmatize.

    Returns:
        list: List of processed tokens.
    """
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize and clean each token
    processed_tokens = [
        lemmatizer.lemmatize(token).lower().strip() 
        for token in tokens
    ]
    
    return processed_tokens


def build_model():
    """Build a machine learning pipeline for text classification.

    Returns:
        Pipeline: A scikit-learn Pipeline object that includes:
            - CountVectorizer: Converts text to a matrix of token counts.
            - TfidfTransformer: Transforms the count matrix to a normalized term-frequency or term-frequency times inverse document-frequency (TF or TF-IDF) representation.
            - MultiOutputClassifier with RandomForestClassifier: Multi-output classifier that uses a RandomForestClassifier as the base estimator.
    """
    # Create a pipeline with text vectorization, TF-IDF transformation, and multi-output classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('starting_verb', StartingVerbExtractor()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Optionally log or print pipeline parameters for debugging
    print("Pipeline parameters:\n", pipeline.get_params())
    
    return pipeline
    

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of a machine learning model on test data.

    Args:
        model (sklearn.pipeline.Pipeline): The model pipeline to evaluate.
        X_test (pandas.Series): Series containing the test messages.
        Y_test (pandas.DataFrame): DataFrame containing the true categories.
        category_names (list): List of category names for evaluation.
    """
    # Predict the categories for the test set
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    # Evaluate and print metrics for each category
    for category in category_names:
        print(f'Category: {category}')
        print(classification_report(Y_test[category], Y_pred_df[category]))
        
        accuracy = accuracy_score(Y_test[category], Y_pred_df[category])
        f1 = f1_score(Y_test[category], Y_pred_df[category], average='weighted')
        precision = precision_score(Y_test[category], Y_pred_df[category], average='weighted')
        recall = recall_score(Y_test[category], Y_pred_df[category], average='weighted')
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print()  # Newline for better readability between categories


def save_model(model, model_filepath):
    """Save a scikit-learn model to a pickle file.

    Args:
        model (sklearn.pipeline.Pipeline): The scikit-learn model to save.
        filepath (str): The path to the file where the model will be saved.
    """
    try:
        # Save the model to a pickle file
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {model_filepath}.")
    except Exception as e:
        print(f"Error saving model to {model_filepath}: {e}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()