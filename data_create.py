from sklearn.model_selection import train_test_split

from src.data.datagen import HeartSoundDataGenerator
from src.utils.config import get_config, update_config
from src.utils.seed import manual_seed
from src.utils.data import *
from src.utils.filter import *
from src.utils.io import *
from src.utils.parser import Parser
from src.utils.signal import *

if __name__ == "__main__":
    cfg = get_config()

    args = Parser().get_args()

    cfg = update_config(cfg, args)

    seed = cfg.get("seed")
    manual_seed(seed)

    save_path = f"{cfg['datagen']['dir']}"
    dataset = cfg["datagen"]["dataset"]
    file_path = f"{save_path}/{dataset}.npz"

    if not dir_exists(save_path):
        create_dir(save_path)
        print(f"Created {save_path}...")
    else:
        if cfg["datagen"]["clean"]:
            if file_exists(file_path):
                delete_file(file_path)
                print(f"Deleted {file_path}...")

    db_path = f"{cfg['datagen']['ds_path']}/{dataset}/"
    generator = HeartSoundDataGenerator(dataset, db_path)
    data = generator.get_data()

    audio_data = np.array(data[0])
    labels = np.array(data[1]).astype(np.float32)

    resampling_rate = cfg["datagen"]["sampling_rate"]

    signals_with_outliers = extract_audio_data(audio_data, resampling_rate)
    signals_with_outliers = extract_signals(signals_with_outliers)
    signals_with_outliers, labels_with_outliers = remove_inconsistent_signals(signals_with_outliers, labels)
    signals_with_outliers = np.array(signals_with_outliers)
    labels_with_outliers = np.array(labels_with_outliers)
    signals_with_outliers = bp_butter_filter(signals_with_outliers, resampling_rate, 2, 500, 20)

    print(f"Data shape with outliers: {signals_with_outliers.shape}")

    raw_features_with_outliers = (signals_with_outliers - np.min(signals_with_outliers)) / (
            np.max(signals_with_outliers) - np.min(signals_with_outliers))

    signals_without_outliers = remove_outliers(signals_with_outliers.copy(), 3)
    signals_without_outliers = bp_butter_filter(signals_without_outliers, resampling_rate, 2, 500, 20)

    print(f"Data shape without outliers: {signals_without_outliers.shape}")

    raw_features_without_outliers = (signals_without_outliers - np.min(signals_without_outliers)) / (
            np.max(signals_without_outliers) - np.min(signals_without_outliers))

    tsplit = cfg["datagen"]["test_split"]

    raw_train_indices_with, raw_test_indices_with = train_test_split(
        np.arange(len(raw_features_with_outliers)),
        test_size=tsplit,
        random_state=seed,
        stratify=labels_with_outliers
    )

    raw_train_features_with = raw_features_with_outliers[raw_train_indices_with]
    raw_train_labels_with = labels_with_outliers[raw_train_indices_with]
    raw_test_features_with = raw_features_with_outliers[raw_test_indices_with]
    raw_test_labels_with = labels_with_outliers[raw_test_indices_with]

    fold = cfg["datagen"]["fold"]

    raw_folds_with = list(skf(fold, raw_train_features_with, raw_train_labels_with, seed))

    raw_train_indices_without, raw_test_indices_without = train_test_split(
        np.arange(len(raw_features_without_outliers)),
        test_size=tsplit,
        random_state=seed,
        stratify=labels_with_outliers
    )

    raw_train_features_without = raw_features_without_outliers[raw_train_indices_without]
    raw_train_labels_without = labels_with_outliers[raw_train_indices_without]
    raw_test_features_without = raw_features_without_outliers[raw_test_indices_without]
    raw_test_labels_without = labels_with_outliers[raw_test_indices_without]

    raw_folds_without = list(skf(fold, raw_train_features_without, raw_train_labels_without, seed))

    smote_features_with, smote_labels_with = smote(raw_features_with_outliers, labels_with_outliers, seed)

    smote_train_indices_with, smote_test_indices_with = train_test_split(
        np.arange(len(smote_features_with)),
        test_size=tsplit,
        random_state=seed,
        stratify=smote_labels_with
    )

    smote_train_features_with = smote_features_with[smote_train_indices_with]
    smote_train_labels_with = smote_labels_with[smote_train_indices_with]
    smote_test_features_with = smote_features_with[smote_test_indices_with]
    smote_test_labels_with = smote_labels_with[smote_test_indices_with]

    smote_folds_with = list(skf(fold, smote_train_features_with, smote_train_labels_with, seed))

    smote_features_without, smote_labels_without = smote(raw_features_without_outliers, labels_with_outliers, seed)

    smote_train_indices_without, smote_test_indices_without = train_test_split(
        np.arange(len(smote_features_without)),
        test_size=tsplit,
        random_state=seed,
        stratify=smote_labels_without
    )

    smote_train_features_without = smote_features_without[smote_train_indices_without]
    smote_train_labels_without = smote_labels_without[smote_train_indices_without]
    smote_test_features_without = smote_features_without[smote_test_indices_without]
    smote_test_labels_without = smote_labels_without[smote_test_indices_without]

    smote_folds_without = list(skf(fold, smote_train_features_without, smote_train_labels_without, seed))

    combined_data = {
        "raw": {
            "with_outliers": {
                "folds": raw_folds_with,
                "features": raw_train_features_with,
                "labels": raw_train_labels_with,
                "test": (raw_test_features_with, raw_test_labels_with)
            },
            "without_outliers": {
                "folds": raw_folds_without,
                "features": raw_train_features_without,
                "labels": raw_train_labels_without,
                "test": (raw_test_features_without, raw_test_labels_without)
            }
        },
        "smote": {
            "with_outliers": {
                "folds": smote_folds_with,
                "features": smote_train_features_with,
                "labels": smote_train_labels_with,
                "test": (smote_test_features_with, smote_test_labels_with)
            },
            "without_outliers": {
                "folds": smote_folds_without,
                "features": smote_train_features_without,
                "labels": smote_train_labels_without,
                "test": (smote_test_features_without, smote_test_labels_without)
            }
        }
    }

    np.savez_compressed(file_path, **combined_data)