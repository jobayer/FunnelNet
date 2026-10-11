import os
from keras import callbacks
from src.utils.io import *


def get_callbacks(save: bool, patience: int, root_dir: str):
    if not save:
        return []

    if not dir_exists(root_dir):
        create_dir(root_dir)

    cbks = [
        callbacks.CSVLogger(
            filename=os.path.join(root_dir, "training_logs.csv"),
            separator=",",
            append=True,
        ),
    ]
    
    if patience != -1:
        cbks.append(
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                verbose=1,
                mode="max",
                restore_best_weights=True,
            )
        )

    return cbks