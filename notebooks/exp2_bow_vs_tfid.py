# bow vs tfidf

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

# Set the MLflow tracking URI
import dagshub
import mlflow
mlflow.set_tracking_uri('https://dagshub.com/manikantmnnit/mini_project_mlops.mlflow') # Set the MLflow tracking URI to the DagsHub project's MLflow server
dagshub.init(repo_owner='manikantmnnit', repo_name='mini_project_mlops', mlflow=True)

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
# Normalize the text data
df = normalize_text(df)
df=df.sample(round(len(df)*0.5), random_state=42, replace=False).reset_index() # Sample the data to reduce computation time

# map sentiment to integer
sentiment={'empty':0,'sadness':1,'enthusiasm':2,'neutral':3,'worry':4,'surprise':5,'love':6,'fun':7,'hate':8,'happiness':9,'boredom':10,'relief':11,'anger':12}
df['sentiment']=df['sentiment'].map(sentiment)

# Set the experiment name
mlflow.set_experiment("Bow vs TfIdf")

# Define feature extraction methods
vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# Define algorithms
algorithms = {
    'LogisticRegression': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
    'MultinomialNB': MultinomialNB(alpha=1.0),  # `alpha` is the Laplace smoothing parameter,
    'XGBoost': XGBClassifier(objective='multi:softmax', num_class=13), # `num_class` is the number of classes
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3) # `learning_rate` is the step size shrinkage used to prevent overfitting
}

# Start the parent run
with mlflow.start_run(run_name="All Experiments") as parent_run:
    # Loop through algorithms and feature extraction methods (Child Runs)
    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                X = vectorizer.fit_transform(df['content'])
                y = df['sentiment']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                

                # Log preprocessing parameters
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.2)
                
                # Model training
                model = algorithm
                model.fit(X_train, y_train)
                
                # Log model parameters
                if algo_name == 'LogisticRegression':
                    mlflow.log_param("C", model.C) # `C` is the inverse of regularization strength
                elif algo_name == 'MultinomialNB':
                    mlflow.log_param("alpha", model.alpha) # `alpha` is the Laplace smoothing parameter
                elif algo_name == 'XGBoost':
                    mlflow.log_param("n_estimators", model.n_estimators) # `n_estimators` is the number of boosting rounds
                    mlflow.log_param("learning_rate", model.learning_rate) # `learning_rate` is the step size shrinkage used to prevent overfitting
                elif algo_name == 'RandomForest':
                    mlflow.log_param("n_estimators", model.n_estimators) # `n_estimators` is the number of trees in the forest
                    mlflow.log_param("max_depth", model.max_depth) # `max_depth` is the maximum depth of the tree
                elif algo_name == 'GradientBoosting':
                    mlflow.log_param("n_estimators", model.n_estimators) # `n_estimators` is the number of boosting rounds
                    mlflow.log_param("learning_rate", model.learning_rate) # `learning_rate` is the step size shrinkage used to prevent overfitting
                    mlflow.log_param("max_depth", model.max_depth) # `max_depth` is the maximum depth of the tree
                
                # Model evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log evaluation metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Save and log the notebook
                mlflow.log_artifact(__file__)
                
                # Print the results for verification
                print(f"Algorithm: {algo_name}, Feature Engineering: {vec_name}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")