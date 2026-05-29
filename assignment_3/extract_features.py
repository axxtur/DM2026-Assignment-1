import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from scipy.stats import skew, kurtosis

def extract_features(df):
    features = {}
    cols = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    
    for col in cols:
        # Basic Statistics
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_median'] = df[col].median()
        features[f'{col}_q25'] = df[col].quantile(0.25)
        features[f'{col}_q75'] = df[col].quantile(0.75)
        features[f'{col}_skew'] = skew(df[col])
        features[f'{col}_kurt'] = kurtosis(df[col])
        
        # Ranges
        features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
        features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
        
    # Magnitude (overall acceleration)
    # Using the mean values to approximate a magnitude per second, then aggregating
    mag = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2)
    features['mag_mean'] = mag.mean()
    features['mag_std'] = mag.std()
    features['mag_max'] = mag.max()
    
    # Simple Temporal: Mean change
    for col in cols:
        features[f'{col}_diff_mean'] = df[col].diff().abs().mean()
        
    return features

def prepare_ml_data(base_path, mode='train'):
    rows = []
    user_dirs = sorted(glob.glob(os.path.join(base_path, mode, mode, "User_*")))
    
    for user_dir in tqdm(user_dirs, desc=f"Extracting features from {mode}"):
        user_id = os.path.basename(user_dir)
        csv_files = sorted(glob.glob(os.path.join(user_dir, "*.csv")))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            f_dict = extract_features(df)
            
            f_dict['file_id'] = df['file_id'].iloc[0]
            f_dict['user_id'] = user_id
            if mode == 'train':
                f_dict['label'] = df['label'].iloc[0]
                
            rows.append(f_dict)
            
    return pd.DataFrame(rows)

if __name__ == "__main__":
    base_path = "assignment_3/nycu-data-mining-assignment-3"
    
    print("Processing Training Features...")
    df_train = prepare_ml_data(base_path, mode='train')
    df_train.to_csv("assignment_3/train_features.csv", index=False)
    
    print("Processing Test Features...")
    df_test = prepare_ml_data(base_path, mode='test')
    df_test.to_csv("assignment_3/test_features.csv", index=False)
    
    print("Feature extraction complete.")
