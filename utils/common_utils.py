import os
import sys
import logging
from tqdm import tqdm
import pickle


def read_txt(file_path, strip_chars=None):
    """
    Read txt file.

    Args:
        file_path: txt file file path.
        strip_chars: strip chars for strip function.
    Return:
        content: list of txt string within given file.
    """
    with open(file_path, mode="r", encoding='utf-8') as file:
        content = [line.strip(strip_chars) for line in file.readlines()]
    return content


def write_txt(file_path, content):
    """
    Write string content to txt file.

    Args:
        file_path: txt file path to write in.
        content: list of string to write.
    Return:
        None.
    """
    with open(file_path, mode="w", encoding='utf-8') as file:
        for line in content:
            file.write(line + "\n")


def folder_files(folder_dir, recursive=False, suffix=None):
    """
    list all files in specific folder.

    Args:
        folder_dir: relative or absolute path to specific folder.
        recursive: whether to recursively collect files in sub-folder.
        prefix: only files end with specific prefix will be collect.
    Return:
        sub_files: all files found in specific folder (exclude directories).
    """
    sub_files = []
    path_list = os.listdir(folder_dir)
    for name in path_list:
        path = os.path.join(folder_dir, name)
        if os.path.isfile(path):
            if suffix is not None and not path.endswith(suffix):
                continue
            sub_files.append(path)
        elif os.path.isdir(path) and recursive:
            sub_files.extend(folder_files(path, recursive, suffix))
    return sub_files


def delete_directory(file_dir):
    """
    Delete the given file or folder.
    Args:
        file_dir: path to file or folder need to delete.
    Return:
        None.
    """
    if os.path.isdir(file_dir):
        path_list = os.listdir(file_dir)
        for name in path_list:
            path = os.path.join(file_dir, name)
            delete_directory(path)
        os.rmdir(file_dir)
    elif os.path.isfile(file_dir):
        os.remove(file_dir)


def save_pickle(file_path, *objects):
    """
    Dump python object(s) to file.

    Args:
        file_path: path to save file.
        objects: list of python objects.
    Return:
        None.
    """
    with open(file_path, "wb") as file:
        for obj in objects:
            pickle.dump(obj, file)


def load_pickle(file_path, items=1):
    """
    Load python object(s) dumped with pickle from file.

    Args:
        file_path: path to ".pkl" file.
        items: number of object to load.
    Return:
        returns: loaded object(s).
    """
    with open(file_path, "rb") as file:
        returns = [pickle.load(file) for _ in range(items)]
    return returns[0] if items == 1 else returns


def get_tqdm(iterable, **kwargs):
    """
    Using tqdm object with "default" configs to wrap the original iterable.
    tqdm refs: https://github.com/tqdm/tqdm#parameters

    Args:
        iterable: an iterable need to traverse.
        **kwargs: extra tqdm config (will overwrite default value).
    """
    configs = {
        "ncols": 80,
        "colour": "#3bfb00",
    }
    configs.update(kwargs)
    return tqdm(iterable, **configs)


def get_logger(name, loglevel=None, logfile=None):
    """
    Get a logger with stdout handler and file handler (if provide logfile arg).

    Args:
        name: logger name.
        loglavel: log level, can be string, integer or none.
        logfile: path of log file.
    Return:
        logger: logging.Logger object.
    """
    
    # get logger and return if logger is already initialized
    logger = logging.getLogger(name)
    if len(logger.handlers) > 0:
        if loglevel is not None or logfile is not None:
            print(f"warning: logger \"{name}\" already exist, extra settings is ignored.")
        return logger
    
    # prevent potential duplicate output
    logger.propagate = False

    # set logger level
    if loglevel is None:
        loglevel = logging.DEBUG
    elif isinstance(loglevel, str):
            loglevel = loglevel.upper()
    logger.setLevel(loglevel)

    # formatter, ref: https://docs.python.org/3/library/logging.html?highlight=logger#logrecord-attributes
    formatter = logging.Formatter("%(asctime)s | <%(name)s> [%(levelname)s]: %(message)s", 
                                  datefmt="%m-%d %H:%M:%S")  # %Y-%m-%d %H:%M:%S
    
    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # logfile handler
    if logfile is not None:
        # ensure existence of parent folder
        dir_path = os.path.dirname(logfile)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        # add file handler
        file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    pass