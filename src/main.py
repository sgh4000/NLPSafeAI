import pandas as pd
import numpy as np
from data import load_data, load_embeddings, load_pca, prepare_data_for_training, get_model
from perturbations import create_perturbations
from hyperrectangles import load_hyperrectangles
from train import train_base, train_adversarial, save_model_in_onnx

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide info/warn logs

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")  # extra safeguard
except Exception:
    pass

# from pca_analysis import pca_graph


if __name__ == '__main__':

    # Load data controller
    # First time running the code, set all to False to generate and save all necessary files
    # Afterwards, set to True to load saved files
    load_saved_embeddings = True
    load_saved_align_mat = True
    load_saved_pca = True
    load_saved_perturbations = True
    load_saved_hyperrectangles = True
    # Control what to run (to save time)
    RUN_BASE_TRAINING = False
    RUN_ADVERSARIAL_TRAINING = False


    # Set up variables which will be used
    n_components = 30
    batch_size = 64
    epochs = 15
    n_classes = 2
    pgd_steps = 5
    epsilon = 0.3
    num_iter = 10
    pos_lable_1_neg_lable_0 = True
    test_train_split_rate = 0.7
    cosine_threshold = 0.2
    seed = 42
    from_logits = True

    # Variables
    path = 'datasets'
    dataset_names = ['depression']
    encoding_models = {'all-MiniLM-L6-v2': 'sbert22M'}
    og_perturbation_name = 'original'
    perturbation_names = ['character']
    hyperrectangles_names = {'character': ['character']}

    # Derived variables
    dataset_name = dataset_names[0]
    encoding_model = list(encoding_models.keys())[0]
    encoding_model_name = encoding_models[encoding_model]
    perturbation_name = perturbation_names[0]
    hyperrectangles_name = list(hyperrectangles_names.keys())[0]
    
    # Load the data and embed them
    data_o = load_data(dataset_name, pos_lable_1_neg_lable_0, test_train_split_rate, path)
    X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o = load_embeddings(dataset_name, encoding_model, encoding_model_name, og_perturbation_name, load_saved_embeddings, load_saved_align_mat, data_o, path)
    print("Data loaded and the embeding is done. Data size: ", len(X_train_pos_embedded_o))

    # Create pthe erturbations and embed them
    data_p = create_perturbations(dataset_name, perturbation_name, data_o, path)
    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p, y_train_pos_p, y_train_neg_p, y_test_pos_p, y_test_neg_p = load_embeddings(dataset_name, encoding_model, encoding_model_name, perturbation_name, load_saved_perturbations, load_saved_align_mat, data=data_p, path=path)
    print("Perturbation data created and the embeding is done. Data size: ", len(X_train_pos_embedded_p))

    # Prepare the data for training
    X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset_name, encoding_model_name, load_saved_pca, X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, n_components, path=path)
    train_dataset, test_dataset = prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o, batch_size)
    print("Data is ready for training. Data size: ", len(train_dataset))

    # Create the hyper-rectangles (needed for adversarial training)
    if RUN_ADVERSARIAL_TRAINING:
        hyperrectangles = load_hyperrectangles(
            dataset_name,
            encoding_model_name,
            hyperrectangles_name,
            load_saved_hyperrectangles,
            epsilon,
            cosine_threshold,
            path=path
        )
        print("Hyper rectangulars are loaded. Hyper rectangular size: ", len(hyperrectangles))


    # ------------------------------------------------------------
    # BASE training
    # ------------------------------------------------------------
    if RUN_BASE_TRAINING:
        model = get_model(n_components)
        model = train_base(model, train_dataset, test_dataset, epochs, seed=seed, from_logits=from_logits)
        save_model_in_onnx(model, "base")
    else:
        print("[INFO] Skipping base training (using existing base.onnx if already saved).")


    # ------------------------------------------------------------
    # ADVERSARIAL training + hyperrectangles
    # ------------------------------------------------------------    
    if RUN_ADVERSARIAL_TRAINING:
        model = get_model(n_components)
        n_samples = int(len(X_train_pos))
        model = train_adversarial(
            model,
            train_dataset,
            test_dataset,
            hyperrectangles,
            epochs,
            batch_size,
            n_samples,
            pgd_steps,
            seed=seed,
            from_logits=from_logits
        )
        save_model_in_onnx(model, "adversarial")
    else:
        print("[INFO] Skipping adversarial training (using existing adversarial.onnx if already saved).")


    # ------------------------------------------------------------
    # SEMANTIC perturbations + hyperrectangles
    # ------------------------------------------------------------
    # 1. Create SEMANTIC perturbations (using semantic_perturbation in perturbations.py)
    # 2. Compute embeddings for these semantic perturbations
    # 3. Build SEMANTIC hyperrectangles, saved as 'semantic.npy'

    semantic_perturbation_name = 'semantic'
    semantic_hyperrectangles_name = 'semantic'

    # 1) Create semantic perturbations
    #    reuse data_o (original data) and just ask for a new perturbation type.
    print("\n[SEMANTIC] Creating semantic perturbations...")
    data_semantic = create_perturbations(
        dataset_name,
        semantic_perturbation_name,
        data_o,
        path=path
    )

    # 2) Compute embeddings for semantic perturbations
    #    set load_saved_embeddings=False the first time so they are generated.
    print("[SEMANTIC] Computing/loading embeddings for semantic perturbations...")
    _ = load_embeddings(
        dataset_name,
        encoding_model,
        encoding_model_name,
        semantic_perturbation_name,
        load_saved_embeddings=False,      # set to False the first time
        load_saved_align_mat=load_saved_align_mat,
        data=data_semantic,
        path=path
    )

    # 3) Build semantic hyperrectangles
    #    hyperrectangles.py will:
    #      - load PCA
    #      - load original + semantic embeddings
    #      - group perturbations by original index
    #      - build min/max boxes in PCA space
    print("[SEMANTIC] Building semantic hyperrectangles...")
    semantic_hyperrectangles = load_hyperrectangles(
        dataset_name,
        encoding_model_name,
        semantic_hyperrectangles_name,
        load_saved_hyperrectangles=False,  # set to False the first time
        eps=epsilon,
        cosine_threshold=cosine_threshold,
        path=path
    )


    try:
        print("[SEMANTIC] Semantic hyperrectangles shape:", semantic_hyperrectangles.shape)
    except Exception:
        print("[SEMANTIC] Semantic hyperrectangles created (shape unknown type).")

    print("[SEMANTIC] Done. Semantic hyperrectangles saved (semantic.npy).")


