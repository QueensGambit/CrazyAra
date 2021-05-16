"""
@file: rl_utils.py
Created on 28.03.2021
@project: crazy_ara
@author: queensgambit, maxalexger

Utility functions to facilitate the reinforcement learning process.
Main purposes: get and change binary name; simple methods to manipulate files & folders.
"""

import re
import os
import glob
import ntpath
import logging


def change_binary_name(binary_dir: str, current_binary_name: str, process_name: str, nn_update_idx: int):
    """
    Change the name of the binary to the process' name (which includes initials,
    binary name and remaining time) & additionally add the current epoch.

    :return: the new binary name
    """
    idx = process_name.find(f'#')
    new_binary_name = f'{process_name[:idx]}_UP={nn_update_idx}{process_name[idx:]}'

    if not os.path.exists(binary_dir + new_binary_name):
        os.rename(binary_dir + current_binary_name, binary_dir + new_binary_name)
        logging.info(f'Changed binary name to: {new_binary_name}')

    return new_binary_name


def create_dir(directory):
    """
    Creates a given director in case it doesn't exists already
    :param directory: Directory path
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory %s" % directory)


def enable_logging(logging_lvl=logging.DEBUG, log_filename=None):
    """
    Enables logging for a given level
    :param logging_lvl: Specifies logging level (e.g. logging.DEBUG, logging.INFO...)
    :param log_filename: Will export all log message into given logfile (append mode) if not None
    :return:
    """
    root = logging.getLogger()
    root.setLevel(logging_lvl)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_lvl)

    formatting_method = "%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s"
    formatter = logging.Formatter(formatting_method, "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_filename is not None:
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def extract_nn_update_idx_from_binary_name(current_binary_name: str) -> int:
    """
    Extract the epoch from our custom build binary name.
    :param current_binary_name: The current name of the binary
    :return: The epoch. If we could not find a match or an error occurred, return 0.
    """
    match = re.search(f'_UP=[0-9]+#', current_binary_name)
    if match:
        try:
            return int(match.group(0)[4:-1])
        except ValueError:
            logging.error(f'Could not find cast the match when extracting epoch from binary name. '
                          f'Returning 0 (which will continue game generation/training).')
            return 0
    else:
        logging.error(f'Could not find a match when extracting epoch from binary name. '
                      f'Returning 0 (which will continue game generation/training).')
        return 0


def get_current_binary_name(binary_dir: str, original_binary_name: str):
    """
    Return the current name of the binary. The binary will be changed by the
    trainer process in order to display training progress.
    :param binary_dir: Directory, where the binary is located.
    :param original_binary_name: The original name of the binary (e.g. CrayAra, MultiAra)
    :return: The current name of the Binary.
    """
    files = glob.glob(f'{binary_dir}*{original_binary_name}*')
    # ntpath.basename() also works on Windows!
    return ntpath.basename(files[0])


def get_log_filename(args, rl_config):
    """
    Returns the file name for the log messages
    :param args: Command line arguments
    :param rl_config: An instance of the RLConfig class from rl_config.py
    :return: Filename: str or None
    """
    if args.export_no_log is False:
        return "%s/%s_%d.log" % (rl_config.binary_dir, args.context, args.device_id)
    return None


def move_all_files(from_dir, to_dir):
    """
    Moves all files from a given directory to a destination directory
    :param from_dir: Origin directory where the files are located
    :param to_dir: Destination directory where all files (including subdirectories directories) will be moved
    :return:
    """
    file_names = os.listdir(from_dir)

    for file_name in file_names:
        os.rename(from_dir + file_name, to_dir + file_name)
