from datetime import datetime, timezone

import os
import numpy as np
from keras import optimizers
from keras.models import load_model

from src.model.model import get_model
from src.utils.callback import get_callbacks
from src.utils.config import get_config, update_config
from src.utils.io import *
from src.utils.parser import Parser
from src.utils.seed import manual_seed
from src.utils.tflite import tf_lite_model

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix


def compute_sensitivity_specificity(y_true, y_pred, n_classes):
    eps = 1e-7
    y_true = np.asarray(y_true)
    if n_classes == 1:
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        y_hat = (y_pred > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        return float(sensitivity), float(specificity)
    else:
        if y_pred.ndim == 1:
            y_hat = y_pred.astype(int)
        else:
            y_hat = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_hat, labels=list(range(n_classes)))
        sensitivities = []
        specificities = []
        for k in range(n_classes):
            tp = cm[k, k]
            fn = cm[k, :].sum() - tp
            fp = cm[:, k].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            sens = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp + eps) if (tn + fp) > 0 else 0.0
            sensitivities.append(sens)
            specificities.append(spec)
        return float(np.mean(sensitivities)), float(np.mean(specificities))

if __name__ == "__main__":
    cfg = get_config()

    args = Parser().get_args()

    cfg = update_config(cfg, args)

    seed = cfg.get("seed")
    manual_seed(seed)

    log_path = cfg["run"]["log_dir"]

    dataset = cfg["run"]["dataset"]

    smote = cfg["run"]["smote"]
    remove_outliers = cfg["run"]["remove_outliers"]

    data_type = "smote" if smote else "raw"
    outlier_type = "without_outliers" if remove_outliers else "with_outliers"

    current_config_name = f"{data_type}__{outlier_type}"

    time_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.join(log_path, dataset, current_config_name, time_now)

    if not dir_exists(log_save_path):
        create_dir(log_save_path)
        print(f"Created {log_save_path}...")

    data_dir = cfg["run"]["data_dir"]

    f_path = os.path.join(data_dir, f"{dataset}.npz")

    if not file_exists(f_path):
        raise FileNotFoundError(f"Dataset file not found: {f_path}")

    full_data = np.load(f_path, allow_pickle=True)

    data = full_data[data_type].item()[outlier_type]

    features = data['features']
    labels = data['labels']

    test_split = cfg.get("run", {}).get("test_split", 0.10)
    all_idx = np.arange(len(labels))
    if test_split and test_split > 0:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_split, stratify=labels, random_state=seed
        )
    else:
        train_idx = all_idx
        test_idx = np.array([], dtype=int)

    n_folds = cfg.get("run", {}).get("fold", 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, va in skf.split(train_idx, labels[train_idx]):
        folds.append((train_idx[tr], train_idx[va]))

    test_indices = test_idx

    unique_values, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for value, count in zip(unique_values, counts):
        print(f"  Class {value}: {count} samples ({count/len(labels):.2%})")
    print(f"Total samples: {len(labels)}")

    i_shape = features.shape[1:]
    unique_labels = np.unique(labels)
    n_classes = 1 if len(unique_labels) == 2 else len(unique_labels)
    loss_fn = ("binary_crossentropy" if n_classes == 1 else "sparse_categorical_crossentropy")

    tmp_model = get_model(i_shape, n_classes)
    trainable_params = np.sum([np.prod(v.shape) for v in tmp_model.trainable_weights])
    
    vrbse = cfg["run"]["verbose"]

    if vrbse:
        tmp_model.summary()

    del tmp_model

    vbs = 1 if vrbse else 0

    opt = cfg["run"]["optim"]
    lr = cfg["run"]["lr"]

    start_time = datetime.now(timezone.utc)
    print(f"Start time: {start_time} (UTC)\n")

    epoch = cfg["run"]["epoch"]
    batch_size = cfg["run"]["batch"]

    best_val_accuracy = 0
    best_val_loss = float('inf')
    best_train_accuracy = 0
    best_train_loss = float('inf')
    best_fold = -1
    best_mdl_save_path = os.path.join(log_save_path, "best_model.keras")

    fold_val_accs = []
    fold_val_losses = []
    fold_train_accs = []
    fold_train_losses = []
    fold_model_paths = []
    fold_val_sensitivities = []
    fold_val_specificities = []

    save_cbks = cfg["run"]["callbacks"]
    early_stopping = cfg["run"]["early_stopping"]

    for i, (train_ids, val_ids) in enumerate(folds):
        print("Training on fold " + str(i + 1) + f"/{len(folds)}...\n")

        Xtrain_data = features[train_ids]
        Xval_data = features[val_ids]
        ytrain_data = labels[train_ids]
        yval_data = labels[val_ids]

        model = get_model(i_shape, n_classes)
        if opt == "adam":
            optim_obj = optimizers.Adam(learning_rate=lr)
        elif opt == "sgd":
            optim_obj = optimizers.SGD(learning_rate=lr)
        elif opt == "rmsprop":
            optim_obj = optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        model.compile(optimizer=optim_obj, loss=loss_fn, metrics=["accuracy"])

        history = model.fit(
            Xtrain_data,
            ytrain_data,
            epochs=epoch,
            verbose=vbs,
            shuffle=True,
            batch_size=batch_size,
            callbacks=get_callbacks(save_cbks, early_stopping, log_save_path),
            validation_data=(Xval_data, yval_data),
        )

        try:
            val_pred_probs = model.predict(Xval_data, batch_size=batch_size, verbose=0)
            val_sens, val_spec = compute_sensitivity_specificity(yval_data, val_pred_probs, n_classes)
        except Exception as e:
            print(f"Warning: could not compute sensitivity/specificity for fold {i+1}: {e}")
            val_sens, val_spec = np.nan, np.nan

        fold_val_sensitivities.append(val_sens)
        fold_val_specificities.append(val_spec)

        fold_val_acc = max(history.history['val_accuracy'])
        fold_val_loss_idx = np.argmax(history.history['val_accuracy'])
        fold_val_loss = history.history['val_loss'][fold_val_loss_idx]

        fold_train_acc = max(history.history['accuracy'])
        fold_train_loss_idx = np.argmax(history.history['accuracy'])
        fold_train_loss = history.history['loss'][fold_train_loss_idx]

        fold_model_path = os.path.join(log_save_path, f"model_fold_{i+1}.keras")
        model.save(fold_model_path)
        fold_model_paths.append(fold_model_path)

        fold_val_accs.append(fold_val_acc)
        fold_val_losses.append(fold_val_loss)
        fold_train_accs.append(fold_train_acc)
        fold_train_losses.append(fold_train_loss)

        if fold_val_acc > best_val_accuracy:
            best_val_accuracy = fold_val_acc
            best_val_loss = fold_val_loss
            best_fold = i
            model.save(best_mdl_save_path)

        if fold_train_acc > best_train_accuracy:
            best_train_accuracy = fold_train_acc
            best_train_loss = fold_train_loss

    end_time = datetime.now(timezone.utc)
    print(f"\nEnd time: {end_time} (UTC)")

    avg_val_accuracy = float(np.mean(fold_val_accs)) if len(fold_val_accs) > 0 else None
    avg_val_loss = float(np.mean(fold_val_losses)) if len(fold_val_losses) > 0 else None
    avg_train_accuracy = float(np.mean(fold_train_accs)) if len(fold_train_accs) > 0 else None
    avg_train_loss = float(np.mean(fold_train_losses)) if len(fold_train_losses) > 0 else None
    avg_val_sensitivity = float(np.nanmean(fold_val_sensitivities)) if len(fold_val_sensitivities) > 0 else None
    avg_val_specificity = float(np.nanmean(fold_val_specificities)) if len(fold_val_specificities) > 0 else None

    if avg_val_accuracy is not None:
        print(f"Avg val acc: {avg_val_accuracy:.5f}")
    print(f"Best val acc: {best_val_accuracy:.5f}")

    test_loss = None
    test_acc = None
    avg_fm_acc = None
    avg_fm_loss = None
    test_sensitivity = None
    test_specificity = None
    avg_fm_sensitivity = None
    avg_fm_specificity = None

    if test_indices.size > 0 and os.path.exists(best_mdl_save_path):
        best_model = load_model(best_mdl_save_path)
        test_loss, test_acc = best_model.evaluate(
        features[test_indices], labels[test_indices], batch_size=batch_size, verbose=1
        )
        print(f"Test set results — Loss: {test_loss:.5f}, Accuracy: {test_acc:.5f}")

        try:
            test_pred_probs = best_model.predict(features[test_indices], batch_size=batch_size, verbose=0)
            test_sensitivity, test_specificity = compute_sensitivity_specificity(labels[test_indices], test_pred_probs, n_classes)
            print(f"Test set sensitivity: {test_sensitivity:.5f}, specificity: {test_specificity:.5f}")
        except Exception as e:
            print(f"Warning: could not compute test sensitivity/specificity: {e}")

        fold_test_accs = []
        fold_test_losses = []
        fold_test_sensitivities = []
        fold_test_specificities = []

        for idx, fm_path in enumerate(fold_model_paths):
            if not os.path.exists(fm_path):
                print(f"Fold model not found: {fm_path} — skipping")
                continue
            fm = load_model(fm_path)
            fm_loss, fm_acc = fm.evaluate(
                features[test_indices], labels[test_indices], batch_size=batch_size, verbose=1
            )
            fold_test_accs.append(fm_acc)
            fold_test_losses.append(fm_loss)
            try:
                fm_pred_probs = fm.predict(features[test_indices], batch_size=batch_size, verbose=0)
                fm_sens, fm_spec = compute_sensitivity_specificity(labels[test_indices], fm_pred_probs, n_classes)
                fold_test_sensitivities.append(fm_sens)
                fold_test_specificities.append(fm_spec)
            except Exception as e:
                print(f"Warning: could not compute sensitivity/specificity for fold model {idx+1}: {e}")

        avg_fm_acc = float(np.mean(fold_test_accs)) if len(fold_test_accs) > 0 else None
        avg_fm_loss = float(np.mean(fold_test_losses)) if len(fold_test_losses) > 0 else None
        avg_fm_sensitivity = float(np.mean(fold_test_sensitivities)) if len(fold_test_sensitivities) > 0 else None
        avg_fm_specificity = float(np.mean(fold_test_specificities)) if len(fold_test_specificities) > 0 else None
    else:
        print("No held-out test set or best model not found — skipping test evaluation")

    result_save_path = os.path.join(log_save_path, "train_results.txt")

    with open(result_save_path, "a") as f:
        f.write(f"Random seed: {seed}\n")
        f.write(f"Start time: {start_time} (UTC)\n")
        f.write(f"End time: {end_time} (UTC)\n")
        f.write(f"Total time: {end_time - start_time}\n")
        f.write(f"SMOTE: {smote}\n")
        f.write(f"Remove outliers: {remove_outliers}\n")
        f.write(f"Epochs: {epoch}\n")
        f.write(f"Optimizer: {opt}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Loss function: {loss_fn}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Fold: {len(folds)}\n")
        f.write(f"Early stopping: {early_stopping}\n")
        f.write(f"Best training accuracy: {best_train_accuracy:.5f}\n")
        f.write(f"Best training loss: {best_train_loss:.5f}\n")
        if avg_train_accuracy is not None:
            f.write(f"Average training accuracy (across folds): {avg_train_accuracy:.5f}\n")
            f.write(f"Average training loss (across folds): {avg_train_loss:.5f}\n")
        f.write(f"Best validation accuracy: {best_val_accuracy:.5f} (fold {best_fold + 1})\n")
        f.write(f"Best validation loss: {best_val_loss:.5f}\n")
        if avg_val_accuracy is not None:
            f.write(f"Average validation accuracy (across folds): {avg_val_accuracy:.5f}\n")
            f.write(f"Average validation loss (across folds): {avg_val_loss:.5f}\n")
        if avg_val_sensitivity is not None:
            f.write(f"Average validation sensitivity (across folds): {avg_val_sensitivity:.5f}\n")
        if avg_val_specificity is not None:
            f.write(f"Average validation specificity (across folds): {avg_val_specificity:.5f}\n")
        if test_acc is not None:
            f.write(f"Best test accuracy: {test_acc:.5f}\n")
            f.write(f"Best test loss: {test_loss:.5f}\n")
        if test_sensitivity is not None:
            f.write(f"Best test sensitivity: {test_sensitivity:.5f}\n")
        if test_specificity is not None:
            f.write(f"Best test specificity: {test_specificity:.5f}\n")
        if avg_fm_acc is not None:
            f.write(f"Average test accuracy (across folds): {avg_fm_acc:.5f}\n")
            f.write(f"Average test loss (across folds): {avg_fm_loss:.5f}\n")
        if avg_fm_sensitivity is not None:
            f.write(f"Average test sensitivity (across folds): {avg_fm_sensitivity:.5f}\n")
        if avg_fm_specificity is not None:
            f.write(f"Average test specificity (across folds): {avg_fm_specificity:.5f}\n")
        f.write(f"Total trainable parameters: {trainable_params}\n")

    print("Saved runtime info...")

    save_tflite = cfg["run"]["tflite"]

    if save_tflite and os.path.exists(best_mdl_save_path):
        saved_tflite_dir = os.path.join(log_save_path, cfg["run"]["tflite_path"])

        if not dir_exists(saved_tflite_dir):
            create_dir(saved_tflite_dir)
        else:
            clear_dir(saved_tflite_dir)

        tflite_model = tf_lite_model(features, best_mdl_save_path)

        tflite_model_name = "model.tflite"

        tflite_path = os.path.join(saved_tflite_dir, tflite_model_name)

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print("Saved TFLite model...")