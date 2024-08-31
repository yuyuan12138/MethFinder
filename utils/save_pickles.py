import os
import shutil
import pickle
from config import config

def delete_all_previous_folder_files(path_list):

    if type(path_list) == str:
        path_list = [path_list]
    for path in path_list:
        if os.path.isdir(path):
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    print(f"Folder {path} and its contents have been deleted.")
                except OSError as e:
                    print(f"Error: {e.strerror}")
        else:
            if os.path.exists(path):
                os.remove(path)
                print(f"File {path} have been deleted.")

def save_umap_pickles(folder_path, umap_analysis, name='ACC'):
    with open(f'{folder_path}/{config.data_name}_best_{name}.pkl', 'wb') as f:
        pickle.dump(umap_analysis, f)