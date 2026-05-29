import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

def load_data(base_path, mode='train'):
    all_data = []
    all_labels = []
    all_file_ids = []
    all_user_ids = []
    
    user_dirs = sorted(glob.glob(os.path.join(base_path, mode, mode, "User_*")))
    
    for user_dir in tqdm(user_dirs, desc=f"Loading {mode} data"):
        user_id = os.path.basename(user_dir)
        csv_files = sorted(glob.glob(os.path.join(user_dir, "*.csv")))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Each file should have 300 rows (0-299 index)
            # We take the 6 features: mean_x, mean_y, mean_z, std_x, std_y, std_z
            features = df[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']].values
            
            if mode == 'train':
                label = df['label'].iloc[0]
                all_labels.append(label)
            
            file_id = df['file_id'].iloc[0]
            
            all_data.append(features)
            all_file_ids.append(file_id)
            all_user_ids.append(user_id)
            
    return np.array(all_data), np.array(all_labels) if mode == 'train' else None, np.array(all_file_ids), np.array(all_user_ids)

if __name__ == "__main__":
    base_path = "assignment_3/nycu-data-mining-assignment-3"
    
    print("Processing Training Data...")
    X_train, y_train, train_file_ids, train_user_ids = load_data(base_path, mode='train')
    np.savez_compressed(os.path.join(base_path, "train_data.npz"), 
                        X=X_train, y=y_train, file_ids=train_file_ids, user_ids=train_user_ids)
    
    print("Processing Test Data...")
    X_test, _, test_file_ids, test_user_ids = load_data(base_path, mode='test')
    np.savez_compressed(os.path.join(base_path, "test_data.npz"), 
                        X=X_test, file_ids=test_file_ids, user_ids=test_user_ids)
    
    print("Data preparation complete.")
