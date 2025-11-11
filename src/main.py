import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from load_data import load_data_kaggle
from sklearn.model_selection import train_test_split
from pca_analysis import pca_graph
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from train import train_base_model
# from save_onnx import save_onnx


if __name__ == '__main__':

    # Set up variables which will be used

    batch_size = 64
    epochs = 6
    n_classes = 2
    epsilon = 0.3
    alpha = 0.1
    num_iter = 10
    file_path = "depression_dataset_reddit_cleaned.csv"
    
    df = load_data_kaggle(file_path)

    depression_count = df['is_depression'].value_counts()
    print(depression_count)

    # Split into train/test
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_depression"]
    )

    # Extract training features and labels if needed. Keeping raw for future analysis
    X_train_raw = df_train["clean_text"].tolist()
    y_train = df_train["is_depression"].values
    X_test_raw = df_test["clean_text"].tolist()
    y_test = df_test["is_depression"].values

    # Optional - PCA analysis
    pca_graph(X_train_raw,X_test_raw,"all-MiniLM-L6-v2")

    encoding_model = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(encoding_model)
    n_components = 30
    batch_size = 64

    X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    data = np.vstack([X_train])
    # PCA data
    data_pca = PCA(n_components=n_components).fit(data)

    X_train = data_pca.transform(X_train)
    X_test = data_pca.transform(X_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    input_size = X_train.shape[1]

    model_base = train_base_model(input_size, train_dataset, test_dataset, epochs)

    loss, acc = model_base.evaluate(X_test, y_test, verbose=0)

    print(f"Base model loss: {loss:.3f}")
    print(f"Base model accuracy: {acc:.3f}")

    # Save base model to onnx file
    # save_onnx(model_base)

    # Save the base model to a keras file

    model_base.save("model_base.h5")










