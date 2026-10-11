import os


def file_exists(path):
    return os.path.isfile(path)


def delete_file(path):
    if file_exists(path):
        os.remove(path)


def dir_exists(path):
    return os.path.isdir(path)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if dir_exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if file_exists(item_path):
                os.remove(item_path)
            elif dir_exists(item_path):
                clear_dir(item_path)
                os.rmdir(item_path)
