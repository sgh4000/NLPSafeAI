from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle as pk
import pandas as pd
import numpy as np
from tensorflow import keras


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide info/warn logs

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")  # extra safeguard
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_model(n_components):
    inputs = keras.Input(shape=(n_components,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def load_data_kaggle(dataset_name, path="datasets"):

    # Load the dataset
    df = pd.read_csv(f'{PROJECT_ROOT}/{path}/{dataset_name}/{dataset_name}.csv')

    return df


def load_data(dataset_name, pos_lable_1_neg_lable_0, test_train_split_rate, path="datasets"):
    # loading the data
    df = load_data_kaggle(dataset_name, path=path)

    # defining positive and negative lable
    if pos_lable_1_neg_lable_0 == True:
        pos_lable = 1
        neg_lable = 0
    else:
        pos_lable = 0
        neg_lable = 1

    # seperate x_neg and x_pos
    x_pos = df[df["is_depression"] == pos_lable].copy()
    x_neg = df[df["is_depression"] == neg_lable].copy()

    # reindex data to being random
    # x_pos = x_pos.sample(frac=1, random_state=42).reset_index(drop=True)
    # x_neg = x_neg.sample(frac=1, random_state=42).reset_index(drop=True)

    train_pos = x_pos.iloc[:(int(len(x_pos)*test_train_split_rate))]
    test_pos = x_pos.iloc[(int(len(x_pos)*test_train_split_rate)):]

    X_train_pos = train_pos['clean_text'].to_numpy()
    X_test_pos = test_pos['clean_text'].to_numpy()
    y_train_pos = train_pos['is_depression'].to_numpy()
    y_test_pos = test_pos['is_depression'].to_numpy()
    

    train_neg = x_neg.iloc[:(int(len(x_neg)*test_train_split_rate))]
    test_neg = x_neg.iloc[(int(len(x_neg)*test_train_split_rate)):]

    X_train_neg = train_neg['clean_text'].to_numpy()
    X_test_neg = test_neg['clean_text'].to_numpy()
    y_train_neg = train_neg['is_depression'].to_numpy()
    y_test_neg = test_neg['is_depression'].to_numpy()
    
    return [X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg]



def load_pca(dataset_name, encoding_model_name, load_saved_pca, X_train_pos, X_train_neg, X_test_pos, X_test_neg,  n_components=30, path='datasets'):
    if load_saved_pca:
        with open(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/pca.pkl', 'rb') as pickle_file:
            data_pca = pk.load(pickle_file)

    else:
        # All data:
        data = np.vstack([X_train_pos, X_train_neg, X_test_pos, X_test_neg])
        # PCA data
        data_pca = PCA(n_components=n_components).fit(data)
        # Save the PCA
        save_path = f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/pca.pkl', 'wb') as pickle_file:
            pk.dump(data_pca, pickle_file)

    X_train_pos = data_pca.transform(X_train_pos)
    X_train_neg = data_pca.transform(X_train_neg)
    X_test_pos = data_pca.transform(X_test_pos)
    X_test_neg = data_pca.transform(X_test_neg)

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg


def prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg, batch_size):
    # Concatenate the pos and neg embeddings and labels
    X_train = np.concatenate((X_train_pos, X_train_neg), axis=0)
    X_test = np.concatenate((X_test_pos, X_test_neg), axis=0)
    y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
    y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    # Get the train and test datasets transformations from tensorflow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def load_align_mat(dataset_name, encoding_model_name, data, load_saved_align_mat, path='datasets'):
    if load_saved_align_mat:
        align_mat = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/align_mat.npy')

    else:
        # Rotate the data, aligning them to the axis
        print(data.shape)
        u, s, vh = np.linalg.svd(a=data)
        align_mat = np.linalg.solve(a=vh, b=np.eye(len(data[0])))

        # Save the alignment matrix
        save_path = f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f'{save_path}/align_mat.npy', align_mat)

    return align_mat

def load_embeddings(dataset_name, encoding_model, encoding_model_name, perturbation_name='original', load_saved_embeddings=None, load_saved_align_mat=None, data=None, path='datasets'):
    if load_saved_embeddings:
        X_train_pos = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/X_train_pos.npy')
        X_train_neg = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/X_train_neg.npy')
        X_test_pos = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/X_test_pos.npy')
        X_test_neg = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/X_test_neg.npy')
        y_train_pos = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/y_train_pos.npy')
        y_train_neg = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/y_train_neg.npy')
        y_test_pos = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/y_test_pos.npy')
        y_test_neg = np.load(f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}/y_test_neg.npy')

    else:
        X_train_pos = data[0]
        X_train_neg = data[1]
        X_test_pos = data[2]
        X_test_neg = data[3]
        y_train_pos = data[4]
        y_train_neg = data[5]
        y_test_pos = data[6]
        y_test_neg = data[7]

        # Embed the sentences
        encoder = SentenceTransformer(f'{encoding_model}')
        X_train_pos = encoder.encode(X_train_pos, show_progress_bar=False)
        X_train_neg = encoder.encode(X_train_neg, show_progress_bar=False)
        X_test_pos = encoder.encode(X_test_pos, show_progress_bar=False)
        X_test_neg = encoder.encode(X_test_neg, show_progress_bar=False)

        # Rotate the data
        align_mat = load_align_mat(dataset_name, encoding_model_name, X_train_pos, load_saved_align_mat, path='datasets')
        X_train_pos = np.matmul(X_train_pos, align_mat)
        X_train_neg = np.matmul(X_train_neg, align_mat)
        X_test_pos = np.matmul(X_test_pos, align_mat)
        X_test_neg = np.matmul(X_test_neg, align_mat)

        # Save the rotated embedded sentences and labels
        save_path = f'{PROJECT_ROOT}/{path}/{dataset_name}/embeddings/{encoding_model_name}/{perturbation_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f'{save_path}/X_train_pos.npy', X_train_pos)
        np.save(f'{save_path}/X_train_neg.npy', X_train_neg)
        np.save(f'{save_path}/X_test_pos.npy', X_test_pos)
        np.save(f'{save_path}/X_test_neg.npy', X_test_neg)
        np.save(f'{save_path}/y_train_pos.npy', y_train_pos)
        np.save(f'{save_path}/y_train_neg.npy', y_train_neg)
        np.save(f'{save_path}/y_test_pos.npy', y_test_pos)
        np.save(f'{save_path}/y_test_neg.npy', y_test_neg)

    return X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos, y_train_neg, y_test_pos, y_test_neg
