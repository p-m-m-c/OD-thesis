# coding: utf-8

# Import and preprocessing of data

import pandas as pd
from scipy.io.arff import loadarff
from scipy.io import loadmat, savemat
import h5py # For version 7.3 of Matlab data

def read_arff_to_df(arff_file, metadata=False):
    """
    Function to convert an arff file located in memory to a pandas DataFrame by using the SciPy module `loadarff`
    
    Input: 
     - arff_file: a valid path to an arff file
     - metadata: Boolean parameter indicating whether metadata has to be saved too
    Output: a pd.DataFrame with the data
    """
    
    data, metadata = loadarff(arff_file) # Function outputs a length-2 tuple
    
    if metadata == False:
        return pd.DataFrame(data), metadata # Return both
    else:
        return pd.DataFrame(data) # Only return the data

def read_matlab_to_df(file_path, feature_prefix='feature_', target_name='target'):
    """
    Function for reading in .mat data formats (including 7.3) and returning the output structured and
    in a pd.DataFrame format, with more regular feature and target names
    """

    # Load the dictionary with all objects
    try:
        mat_dict = loadmat(file_name=file_path) # For all versions <= 7.2
        features = pd.DataFrame(mat_dict['X'], columns=[feature_prefix+str(i) for i in range(mat_dict['X'].shape[1])])
        target = pd.DataFrame(mat_dict['y'], columns=[target_name], dtype=int)

    # I.e. for versions > 7.3
    except NotImplementedError:
        with h5py.File(file_path, 'r') as f:
            features = pd.DataFrame(list(f['X'])).T
            features.columns = [feature_prefix+str(i) for i in range(features.shape[1])] # Generate column names of feature_0, ..., feature_n-1
            target = pd.DataFrame(list(f['y'][0]), columns=[target_name], dtype=int)

    # Concat target and features
    df = pd.concat([features, target], axis=1)

    return df


aloi_df = read_arff_to_df(' ~/Documents/code-data/raw-data/ALOI/ALOI_withoutdupl_norm.arff') # Binary string as label
wilt_df = read_arff_to_df(' ~/Documents/code-data/raw-data/Wilt/Wilt_withoutdupl_norm_05.arff') # Binary string as label
pima_df = read_arff_to_df(' ~/Documents/code-data/raw-data/Pima/Pima_withoutdupl_norm_35.arff') # Binary string as label
thyroid_df = read_arff_to_df(' ~/Documents/code-data/raw-data/Annthyroid/Annthyroid_withoutdupl_norm_07.arff') # Binary string as label
waveform_01_df = read_arff_to_df(' ~/Documents/code-data/raw-data/Waveform/Waveform_withoutdupl_norm_v01.arff') # Binary string as label
waveform_03_df = read_arff_to_df(' ~/Documents/code-data/raw-data/Waveform/Waveform_withoutdupl_norm_v03.arff') # Binary string as label
smtp_df = read_arff_to_df(' ~/Documents/code-data/preprocessed-data/smtp_withoutdupl_norm.arff')
http_df = read_arff_to_df(' ~/Documents/code-data/preprocessed-data/http_withoutdupl_norm.arff')
forest_df = read_arff_to_df(' ~/Documents/code-data/preprocessed-data/ForestCover_withoutdupl_norm.arff')


# Preprocessing of data retrieved from [Rayana](http://odds.cs.stonybrook.edu)

def normalize_features(input_df):
    """
    Function to normalize features in datasets where that is still necessary: scaled to [0,1] with sklearn.MinMaxScaler() object.
    
    Input:
     - input_df: pd.DataFrame read by `read_arff_to_df` or `read_matlab_to_df`, or any other DataFrame
     
    Output:
     - pd.DataFrame with scaled values
    """
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Define feature names to assign them back after scaling or into arff output file
    feature_names = input_df.columns
    
    # Instantiate MinMaxScaler object to map features into [0,1]
    min_max_scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(min_max_scaler.fit_transform(input_df), columns = feature_names)
    
    # If no path is provided
    return scaled_df

def write_df_to_arff(input_df, path_to_write_location):
    """
    Function to write a pd DataFrame to disk in .arff format (for consistency among datasets).
    
    Input: path_to_write_location, a valid absolute path where to write the data to
    Output: None
    """
    # If file has to be saved externally: scale and write to disk
    import arff
    
    feature_names = input_df.columns
    arff.dump(path_to_write_location, input_df.values, relation='df', names=feature_names)
    print("ARFF written to {}".format(path_to_write_location))
    return None


normalize_features(forest_df, ' ~/Documents/code-data/data/ForestCover_withoutdupl_norm.arff') # Norm and write forestcover dataset
normalize_features(smtp_df, ' ~/Documents/code-data/data/smtp_withoutdupl_norm.arff') # Norm and write smtp dataset
normalize_features(http_df, ' ~/Documents/code-data/data/http_withoutdupl_norm.arff') # Norm and write http dataset


# Check for duplicates and modify target to {0 -> inlier, 1 -> outlier}

# Preprocessing steps taken:
# - Read in each dataset
# - Check for duplicates (and remove those if necessary)
# - Modify targets into binary integer column called outlier
# - Write to .arff and .mat

# Write all data to arff format

for file in [filename for filename in os.listdir('../preprocessed-data/') if not filename.startswith('.')]: # Skip the .DS_store cache files
    df = read_arff_to_df(f'../preprocessed-data/{file}')
    savemat(f'../preprocessed-data/{file.rstrip(".arff")}', df.to_dict('list')) # Save as .mat files

for file in os.listdir('../preprocessed-data/'):
    dataset_name = file.split('_')[0]
    data = read_arff_to_df(f'../preprocessed-data/{file}')
    print(f'Dataset {dataset_name}:', '\n', f'perc outliers: {data["outlier"].value_counts().min() / data["outlier"].value_counts().max()}')

