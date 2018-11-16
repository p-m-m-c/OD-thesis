# coding: utf-8

# Import standard libraries
import numpy as np
import pandas as pd
import time
import os

# Import models from PyOD library for outlier detection
from pyod.models.knn import KNN
from pyod.models.lof import LOF

# Import metrics 
from sklearn.metrics import roc_curve, roc_auc_score, precision_score
from pyod.utils import precision_n_scores
import timeit

# Import plotly for visual evaluation
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(False)


# Function definitions
def read_arff_to_df(arff_file, metadata=False):
    """
    Function to convert an arff file located in memory to a pandas DataFrame by using the SciPy module `loadarff`
    
    Input: 
     - arff_file: a valid path to an arff file
     - metadata: Boolean parameter indicating whether metadata has to be saved too
    Output: a pd.DataFrame with the data
    """
    
    # Import the core code
    from scipy.io.arff import loadarff
    
    data, metadata = loadarff(arff_file) # Function outputs a length-2 tuple
    
    if metadata == False:
        return pd.DataFrame(data), metadata # Return both
    else:
        return pd.DataFrame(data) # Only return the data

def normalize_1d_array(arr):
    """
    Function to normalize a 1d array (with outlier scores) to a [0, 1] interval, using the min and the max. 
    
    Input: array, 1d with raw outlier scores
    Returns: array, with values within 0 and 1 interval
    """
    
    # Extract min and max value from the array
    min_value, max_value = arr.min(), arr.max()
    value_range = max_value - min_value
    
    # Scale and return the array
    return np.array([(value - min_value) / value_range for value in arr])

def run_LOF_base_detector(data, k, metric='euclidean', p=2):
    """
    Function to fit and predict the LOF base detector on `data`.
    
    Input:
     - data: pd.DataFrame, to run LOF on
     - k: integer, parameter to indicate the amount of neighbours to include in relative density determination
     - metric: string, distance metric to use, default `euclidean`
     - p: int, default 1 since metric = `euclidean`, otherwise set according to distance metric
     
    Output:
     - clf of class pyod.models.lof.LOF with all its properties
    """
    
    # Split data in values and targets: some datasets have an ID column, others don't
    try:
        X = data.drop(['outlier', 'id'], axis=1)
    except KeyError:
        X = data.drop('outlier', axis=1)
    
    # Construct and fit classifier
    clf = LOF(n_neighbors=k, metric='euclidean', p=p)
    clf.fit(X) # Fit only on features
    
    # Add ground truth labels for evaluation of the classifier
    clf.true_labels_ = data['outlier']
    
    # Return the classifier for further processing
    return clf

def run_KNN_base_detector(data, k, metric='euclidean', p=2, method='mean'):
    """
    Function to fit and predict the KNN base detector on `data`.
    
    Input:
     - data: pd.DataFrame, to run KNN on
     - k: integer, parameter to indicate the amount of neighbours to include in relative density determination
     - metric: string, distance metric to use, default `euclidean`
     - p: int, default 2 since metric = `euclidean`, otherwise set according to distance metric
     
    Output:
     - clf of class pyod.models.knn.KNN with all its properties
    """
    
    # Split data in values and targets: some datasets have an ID column, others don't
    try:
        X = data.drop(['outlier', 'id'], axis=1)
    except KeyError:
        X = data.drop('outlier', axis=1)
    
    # Construct and fit classifier
    clf = KNN(n_neighbors=k, metric='euclidean', p=p, method=method)
    clf.fit(X) # Fit only on features
    
    # Add ground truth labels for evaluation of the classifier
    clf.true_labels_ = data['outlier']
    
    # Return the classifier for further processing
    return clf

# Add one row to specific dataset

def iterative_base_det(data, base_detector, pct_k):
    """
    Function to run a new OD base detector and return a dict that gets used as a row in a DataFrame
    
    Input:
     - Data: pd.Series, the column data to write to the file in `file_path`
     
    Output:
     - A dict entry with Model, the percentage of neighbours, ROC AUC and AP
    """
    
    # Determine the number of neighbours based on the data
    n_neighbours = round((pct_k/100) * len(data))
    
    # Instantiate a new, empty result dict 
    result_dict = {}

    # Call and fit the base detector with the specified parameters
    if base_detector == 'LOF':
        fitted_clf = run_LOF_base_detector(data=data, k=n_neighbours)

    # Call and fit the base detector with the specified parameters    
    elif base_detector == 'KNN':
        fitted_clf = run_KNN_base_detector(data=data, k=n_neighbours)

    # Normalize decision scores for understandability
    fitted_clf.norm_decision_scores_ = normalize_1d_array(fitted_clf.decision_scores_)

    # Calc evaluation metrics based on true labels and normalized decision scores
    roc = np.round(roc_auc_score(fitted_clf.true_labels_, fitted_clf.norm_decision_scores_), decimals=4) # ROC AUC score
    prn = np.round(precision_score(fitted_clf.true_labels_, fitted_clf.labels_), decimals=4) # SKlearn average precision (AP)
    p_at_k = np.round(precision_n_scores(fitted_clf.true_labels_, fitted_clf.norm_decision_scores_), 4) # PyOD P@K score

    # Save scores in result dict
    result_dict = {'Model': base_detector,
                   'K': pct_k,
                   'ROC AUC': roc,
                   'AP': prn,
                   'P at k': p_at_k}
    
    return result_dict

def add_result_to_existing_csv(file_path, result_run):
    """
     Input:
     - File_path: str, a valid absolute file path leading to the place where the file is stored. If it does not exist yet, it creates one.
     - Result_run: dict, the output value of `iterative_base_det` function, so a dict with model -> {LOF, KNN}, K -> {1, ..., 10}, etc.
     
     Output:
     - None, a csv is appended or written
    """
    
    # Read in the existing file or create a new file
    try:
        df = pd.read_csv(file_path, index_col=0)
        next_ = len(df) # The index to insert the next dictionary
        df.loc[next_] = result_run # Insert a new row to the DataFrame
    except:
        df = pd.DataFrame.from_dict(result_run, orient='index').T
    
    # Write to csv
    df.to_csv(file_path)

file = 'filename.arff'
percentages_neighbours = [1, 3, 5, 10, 15, 20, 25, 35] # List with percentage n_neighbours
for pct in percentages_neighbours:
    dataset_name = file.split('_')[0] # Extract name from path to data
    data = read_arff_to_df(f'../preprocessed-data/{file}')
    result_LOF = iterative_base_det(data, 'LOF', pct) # Get one-run result for LOF
    add_result_to_existing_csv(f'~/Documents/code-data/results/second-run/base-detectors/variable-k/{dataset_name}-LOF.csv', result_LOF)
    result_KNN = iterative_base_det(data, 'KNN', pct) # Get one-run result for KNN
    add_result_to_existing_csv(f'~/Documents/code-data/results/second-run/base-detectors/variable-k/{dataset_name}-KNN.csv', result_KNN)
    print(f'Percentage {pct} done')