import argparse
from .config import get_config


class Parser:

    def __init__(self) -> None:
        self._config = get_config()
        self._parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    def get_args(self) -> dict:
        known_args, unknown_args = self._parser.parse_known_args()
        args_dict = vars(known_args)

        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                key = arg[2:]

                if "=" in key:
                    key, value = key.split("=", 1)
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith(
                        "--"
                ):
                    value = unknown_args[i + 1]
                    i += 1
                else:
                    value = True

                if isinstance(value, str):
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass

                args_dict[key] = value
            i += 1

        return args_dict
