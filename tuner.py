from datetime import datetime, timezone

import keras_tuner as kt
import numpy as np
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from src.model.model import get_model
from src.utils.config import get_config, update_config
from src.utils.io import *
from src.utils.parser import Parser
from src.utils.seed import manual_seed

if __name__ == "__main__":
    cfg = get_config()
    args = Parser().get_args()
    cfg = update_config(cfg, args)
    seed = cfg.get("seed")
    manual_seed(seed)
    log_path = cfg["tuning"]["save_path"]

    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_path, datetime_str)

    if not dir_exists(log_path):
        create_dir(log_path)

    if cfg["tuning"]["clean"]:
        if dir_exists(log_path):
            clear_dir(log_path)
            print(f"Cleaned {log_path}...")

    if not dir_exists(log_path):
        create_dir(log_path)
        print(f"Created {log_path}...")

    data_dir = cfg["tuning"]["data_dir"]
    dataset = cfg["tuning"]["dataset"]

    smote = cfg["tuning"]["smote"]
    remove_outliers = cfg["tuning"]["remove_outliers"]

    f_path = f"{data_dir}/{dataset}.npz"

    if not file_exists(f_path):
        raise FileNotFoundError(f"Dataset file not found: {f_path}")

    full_data = np.load(f_path, allow_pickle=True)

    data_type = "smote" if smote else "raw"
    outlier_type = "without_outliers" if remove_outliers else "with_outliers"

    data = full_data[data_type].item()[outlier_type]

    features = data["features"]
    labels = data["labels"]
    folds = data["folds"]

    i_shape = features.shape[1:]
    unique_labels = np.unique(labels)
    n_classes = 1 if len(unique_labels) == 2 else len(unique_labels)
    loss_fn = (
        "binary_crossentropy" if n_classes == 1 else "sparse_categorical_crossentropy"
    )

    first_fold = folds[0]
    train_ids, val_ids = first_fold
    Xtrain_data = features[train_ids]
    Xval_data = features[val_ids]
    ytrain_data = labels[train_ids]
    yval_data = labels[val_ids]

    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            optimizer = None
            model = get_model(i_shape, n_classes)
            optimizer_name = hp.Choice("optimizer", ["adam", "adamw", "sgd", "rmsprop"])

            learning_rate = hp.Float(
                "learning_rate", min_value=0.0001, max_value=0.01, sampling="log"
            )

            if optimizer_name == "adam":
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == "adamw":
                optimizer = optimizers.AdamW(learning_rate=learning_rate)
            elif optimizer_name == "sgd":
                optimizer = optimizers.SGD(learning_rate=learning_rate)
            elif optimizer_name == "rmsprop":
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

            return model

        def fit(self, hp, model, *args_, **kwargs):
            batch_size = hp.Int("batch_size", 32, 128, step=32)
            return model.fit(*args_, batch_size=batch_size, **kwargs)

    trial = cfg["tuning"]["trial"]

    tuner = kt.BayesianOptimization(
        MyHyperModel(),
        objective="val_accuracy",
        max_trials=trial,
        directory=log_path,
        project_name="hyperparam_tuning",
    )

    start_time = datetime.now(timezone.utc)
    print(f"Start time: {start_time} (UTC)\n")
    print("Performing hyperparameter tuning on the first fold...")

    tuner.search(
        Xtrain_data,
        ytrain_data,
        epochs=cfg["tuning"]["max_epochs"],
        validation_data=(Xval_data, yval_data),
    )

    best_hp = tuner.get_best_hyperparameters()[0]

    best_model = MyHyperModel().build(best_hp)

    max_epoch = cfg["tuning"]["max_epochs"]
    early_stopping_patience = cfg["tuning"]["early_stopping"]

    history = best_model.fit(
        Xtrain_data,
        ytrain_data,
        epochs=max_epoch,
        batch_size=best_hp["batch_size"],
        validation_data=(Xval_data, yval_data),
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                patience=early_stopping_patience,
                restore_best_weights=False,
            )
        ],
        verbose=2,
    )

    best_configs = {
        "optimizer": best_hp["optimizer"],
        "learning_rate": best_hp["learning_rate"],
        "batch_size": best_hp["batch_size"],
    }

    end_time = datetime.now(timezone.utc)
    print(f"\nEnd time: {end_time} (UTC)")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.5f}")

    if cfg["tuning"]["record"]:
        with open(f"{log_path}/best_hyperparameters.txt", "w") as f:
            f.write(f"Random seed: {seed}\n")
            f.write(f"Start time: {start_time} (UTC)\n")
            f.write(f"End time: {end_time} (UTC)\n")
            f.write(f"Total time: {end_time - start_time}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Data type: {data_type}\n")
            f.write(f"SMOTE: {smote}\n")
            f.write(f"Outlier removal: {remove_outliers}\n")
            f.write(f"Trials: {trial}\n")
            f.write(f"Max epochs: {max_epoch}\n")
            f.write(f"Early stopping patience: {early_stopping_patience}\n")
            f.write(f"Outlier type: {outlier_type}\n")
            for key, value in best_configs.items():
                f.write(f"{key}: {value}\n")
            f.write(
                f"Best validation accuracy: {max(history.history['val_accuracy']):.5f}\n"
            )
        print("Saved best hyperparameters...")
