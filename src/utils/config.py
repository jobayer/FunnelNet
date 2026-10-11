from yaml import safe_load

from .. import const


def update_config(cfg, args):
    if not cfg or not args:
        return cfg

    if hasattr(args, "__dict__"):
        args = vars(args)

    def _update_recursive(config, _k, _v):
        if "." in _k:
            main_key, rest_key = _k.split(".", 1)
            if main_key in config and isinstance(config[main_key], dict):
                config[main_key] = _update_recursive(config[main_key], rest_key, _v)
        else:
            if _k in config:
                config[_k] = _v
            else:
                config[_k] = _v
        return config

    for key, value in args.items():
        if value is not None:
            cfg = _update_recursive(cfg, key, value)

    return cfg


def get_config():
    try:
        with open(const.CONFIG_FILE, "r") as file:
            yaml_file = safe_load(file)
            return yaml_file
    except FileNotFoundError:
        print(f"Error: Config file '{const.CONFIG_FILE}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the config file: {e}")
        return None
