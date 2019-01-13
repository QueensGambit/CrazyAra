"""
@file: Colorer.py

Script which allows are colored logging output multiplattform.
The script is based on this post and was slightly adjusted:
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

'Here is a solution that should work on any platform. If it doesn't just tell me and I will update it.

How it works: on platform supporting ANSI escapes is using them (non-Windows) and on Windows
it does use API calls to change the console colors.

The script does hack the logging.StreamHandler.emit method from standard library adding a wrapper to it.'

by Sorin & Dave
"""

import logging
import sys
import platform


def add_coloring_to_emit_windows(fn):
    # patch Python code to add color support to logging.StreamHandler

    # add methods we need to the class
    def _out_handle(self):
        import ctypes

        return ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)

    # noinspection PyUnusedLocal
    out_handle = property(_out_handle)

    def _set_color(self, code):
        import ctypes

        # Constants from the Windows API
        self.STD_OUTPUT_HANDLE = -11
        hdl = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
        ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code)

    setattr(logging.StreamHandler, "_set_color", _set_color)

    # noinspection PyPep8Naming,PyUnusedLocal
    def new(*args):
        foreground_blue = 0x0001  # text color contains blue.
        foreground_green = 0x0002  # text color contains green.
        foreground_red = 0x0004  # text color contains red.
        foreground_intensity = 0x0008  # text color is intensified.
        foreground_white = foreground_blue | foreground_green | foreground_red
        # winbase.h
        std_input_handle = -10
        std_output_handle = -11
        std_error_handle = -12

        # wincon.h
        foreground_black = 0x0000
        foreground_blue = 0x0001
        foreground_green = 0x0002
        foreground_cyan = 0x0003
        foreground_red = 0x0004
        foreground_magenta = 0x0005
        foreground_yellow = 0x0006
        foreground_gray = 0x0007
        foreground_intensity = 0x0008  # foreground color is intensified.

        background_black = 0x0000
        background_blue = 0x0010
        background_green = 0x0020
        background_cyan = 0x0030
        background_red = 0x0040
        background_magenta = 0x0050
        background_yellow = 0x0060
        background_gray = 0x0070
        background_intensity = 0x0080  # background color is intensified.

        levelno = args[1].levelno
        if levelno >= 50:
            color = background_yellow | foreground_red | foreground_intensity | background_intensity
        elif levelno >= 40:
            color = foreground_red | foreground_intensity
        elif levelno >= 30:
            color = foreground_yellow | foreground_intensity
        elif levelno >= 20:
            color = foreground_green
        elif levelno >= 10:
            color = foreground_magenta
        else:
            color = foreground_white
        # noinspection PyProtectedMember
        args[0]._set_color(color)

        ret = fn(*args)
        # noinspection PyProtectedMember
        args[0]._set_color(foreground_white)
        # print "after"
        return ret

    return new


def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        level_number = args[1].levelno
        if level_number >= 50:
            color = "\x1b[31m"  # red
        elif level_number >= 40:
            color = "\x1b[31m"  # red
        elif level_number >= 30:
            color = "\x1b[33m"  # yellow
        elif level_number >= 20:
            color = "\x1b[94m"  # light blue
        elif level_number >= 10:
            color = "\x1b[32m"  # green
            # color = '\x1b[90m' # bright black
            #
            # #'\x1b[35m' # pink
        else:
            color = "\x1b[0m"  # normal
        args[1].msg = color + args[1].msg + "\x1b[0m"  # normal
        # print "after"
        return fn(*args)

    return new


def enable_color_logging(debug_lvl=logging.DEBUG):
    if platform.system() == "Windows":
        # Windows does not support ANSI escapes and we are using API calls to set the console color
        logging.StreamHandler.emit = add_coloring_to_emit_windows(logging.StreamHandler.emit)
    else:
        # all non-Windows platforms are supporting ANSI escapes so we use them
        logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)

    root = logging.getLogger()
    root.setLevel(debug_lvl)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(debug_lvl)
    # FORMAT = '[%(asctime)-s][%(name)-s][\033[1m%(levelname)-7s\033[0m] %(message)-s'
    # FORMAT='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

    # FORMAT from https://github.com/xolox/python-coloredlogs
    formatting_method = "%(asctime)s %(name)s[%(process)d] \033[1m%(levelname)s\033[0m %(message)s"

    # FORMAT="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    formatter = logging.Formatter(formatting_method, "%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
