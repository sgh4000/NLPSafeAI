########################################################
### I want to do some analysis on the PCA embeddings ###
########################################################

# Also, I want to investigate how different number of components has an effect

# The pca.explained_variance_ratio returns a vector of the variance explained by each dimension
# So [0.11,0.095,0.085] means that the first dimension explains 11% of variance and so on
# explained_variance_ratio_.cumsum gives the cumulated totals, so taking the final value gives the
# total variance explained by all dimensions

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from sklearn.decomposition import PCA

def pca_graph(X_train_raw, X_test_raw, encoding_model):

    encoder = SentenceTransformer(encoding_model)
    n_components = 30

    X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    data = np.vstack([X_train])
    # PCA data
    data_pca = PCA(n_components=n_components).fit(data)

    print("---- n = 30 ----")

    cum_sum_30 = data_pca.explained_variance_ratio_.cumsum()[-1]

    print(f"Cum sum {n_components} is {cum_sum_30:.3f}")

    n_components = 50

    X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    data = np.vstack([X_train])
    # PCA data
    data_pca = PCA(n_components=n_components).fit(data)

    print("---- n = 50 ----")

    cum_sum_50 = data_pca.explained_variance_ratio_.cumsum()[-1]    

    print(f"Cum sum {n_components} is {cum_sum_50:.3f}")


    # n_components = 70

    # X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    # X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    # data = np.vstack([X_train])
    # PCA data
    # data_pca = PCA(n_components=n_components).fit(data)

    # print("---- n = 70 ----")

    # cum_sum_70 = data_pca.explained_variance_ratio_.cumsum()[-1]

    # print(f"Cum sum {n_components} is {cum_sum_70:.3f}")

    # n_components = 100

    # X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    # X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    # data = np.vstack([X_train])
    # PCA data
    # data_pca = PCA(n_components=n_components).fit(data)

    # print("---- n = 100 ----")

    # cum_sum_100 = data_pca.explained_variance_ratio_.cumsum()[-1]

    # print(f"Cum sum {n_components} is {cum_sum_100:.3f}")

    # n_components = 200

    # X_train = encoder.encode(X_train_raw, show_progress_bar=False)
    # X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    # data = np.vstack([X_train])
    # PCA data
    # data_pca = PCA(n_components=n_components).fit(data)

    # print("---- n = 200 ----")

    # cum_sum_200 = data_pca.explained_variance_ratio_.cumsum()[-1]

    # print(f"Cum sum {n_components} is {cum_sum_200:.3f}")

    # plt.plot([30, 50, 70, 100, 200], [cum_sum_30, cum_sum_50, cum_sum_70, cum_sum_100, cum_sum_200], marker='o')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance Ratio')
    # plt.title('Cumulative Explained Variance Ratio by Principal Components')
    # plt.ylim(bottom=0)
    # plt.show()