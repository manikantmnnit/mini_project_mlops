# hyperparameter tuning

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/campusx-official/mlops-mini-project.mlflow')
dagshub.init(repo_owner='campusx-official', repo_name='mlops-mini-project', mlflow=True)

# Load the data
# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])


# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# Normalize the text data
df = normalize_text(df)

# map sentiment to integer
sentiment={'empty':0,'sadness':1,'enthusiasm':2,'neutral':3,'worry':4,'surprise':5,'love':6,'fun':7,'hate':8,'happiness':9,'boredom':10,'relief':11,'anger':12}
df['sentiment']=df['sentiment'].map(sentiment)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content']) # Convert text data to numbers
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data

# Set the experiment name
mlflow.set_experiment("LoR Hyperparameter Tuning") # Set the experiment name

# Define hyperparameter grid for Logistic Regression
param_grid = {   # Define hyperparameter grid for Logistic Regression
    'C': [0.1, 1, 10],  # Regularization parameter (smaller values specify stronger regularization)
    'penalty': ['l1', 'l2'],  # Regularization type (l1: Lasso, l2: Ridge)
    'max_iter': [100, 200]  # Maximum number of iterations to converge
    }

# Start the parent run for hyperparameter tuning
with mlflow.start_run():

    # Perform grid search
    grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), param_grid, cv=5, scoring='f1', n_jobs=-1) # Perform grid search with 5-fold cross-validation and F1 score as the evaluation metric and use all available CPU cores for parallel processing
    grid_search.fit(X_train, y_train) # Fit the model   

    # Log each parameter combination as a child run
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
            model = LogisticRegression(**params)
            model.fit(X_train, y_train) # Train the model
            
            # Model evaluation
            y_pred = model.predict(X_test)  # Make predictions
            accuracy = accuracy_score(y_test, y_pred, normalize=True) # Calculate accuracy
            precision = precision_score(y_test, y_pred, average='weighted') # Calculate precision
            recall = recall_score(y_test, y_pred, average='weighted')   # Calculate recall
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log parameters and metrics
            mlflow.log_params(params) # Log hyperparameters
            mlflow.log_metric("mean_cv_score", mean_score)  # Log mean CV score
            mlflow.log_metric("std_cv_score", std_score)    # Log standard deviation of CV score
            mlflow.log_metric("accuracy", accuracy)        # Log accuracy
            mlflow.log_metric("precision", precision)   # Log precision
            mlflow.log_metric("recall", recall)     # Log recall 
            mlflow.log_metric("f1_score", f1)       # Log F1 score
            
            
            # Print the results for verification
            print(f"Mean CV Score: {mean_score}, Std CV Score: {std_score}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

    # Log the best run details in the parent run
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)
    
    print(f"Best Params: {best_params}")
    print(f"Best F1 Score: {best_score}")

    # Save and log the notebook
    mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")