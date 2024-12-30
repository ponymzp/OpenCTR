import os
import kagglehub
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

def get_criteo_data(data_type='sample'):

    nrows = None
    if data_type == 'sample':
        nrows = 100000

    file_path = "~/.cache/kagglehub/datasets/mrkmakr/criteo-dataset/versions/1"
    file_path = os.path.expanduser(file_path)

    if not os.path.exists(file_path):
        file_path = kagglehub.dataset_download(
            "mrkmakr/criteo-dataset"
        )

    print("Path to criteo dataset files:", file_path)

    dense_cols = [f'I{i}' for i in range(1, 14)]
    sparse_cols = [f"C{i}" for i in range(1, 27)]
    data_cols = ['Label'] + dense_cols + sparse_cols

    data = pd.read_csv(
        '{}/dac/train.txt'.format(file_path),
        sep='\t',
        header=None,
        names=data_cols,
        nrows=nrows
    )

    for col in data.columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
        if col[0] == 'C':
            unique_classes = data[col].unique()
            random_mapping = np.random.choice(range(len(unique_classes)), size=len(unique_classes), replace=False)
            mapping_dict = dict(zip(unique_classes, random_mapping))
            data[col] = data[col].map(mapping_dict)


    y = data['Label']
    X = data.drop(columns=['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    get_criteo_data()



