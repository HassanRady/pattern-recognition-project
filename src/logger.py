import logging
from logging import handlers
import sys
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent

FORMATTER = logging.Formatter(
    "%(asctime)s|%(levelname)s|%(funcName)s|%(lineno)d: %(message)s"
)


def get_console_handler() -> logging.StreamHandler:
    """
    Creates and returns a console handler for logging.

    This function initializes a logging handler that outputs log messages to the console (stdout)
    with a predefined formatter.

    Returns:
    -------
    logging.StreamHandler
        A console stream handler configured with the specified formatter.

    Notes:
    ------
    - Assumes a global `FORMATTER` object is defined to format log messages.
    - Directs log messages to `sys.stdout`.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(path: str) -> logging.handlers.TimedRotatingFileHandler:
    """
    Creates and returns a timed rotating file handler for logging.

    This function initializes a logging handler that writes log messages to a file
    in a directory specified by `path`. The logs are rotated based on time intervals.

    Parameters:
    ----------
    path : str
        The full file path for the log file, including the file name. The parent
        directories are created if they do not already exist.

    Returns:
    -------
    logging.handlers.TimedRotatingFileHandler
        A file handler configured for timed log rotation.

    Notes:
    ------
    - The log file is rotated daily at midnight.
    - Assumes a global `FORMATTER` object is defined to format log messages.
    """
    logs_path: Path = Path(path)
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(logs_path)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_socket_handler(
    host: Optional[str] = "localhost", port: Optional[int] = 9999
) -> handlers.SocketHandler:
    """
    Creates and returns a socket handler for logging.

    This function initializes a logging handler that sends log messages over a socket
    to a specified host and port. The handler is configured with a predefined formatter.

    Parameters:
    ----------
    host : Optional[str], default="localhost"
        The hostname or IP address of the logging server.
    port : Optional[int], default=9999
        The port number of the logging server.

    Returns:
    -------
    logging.handlers.SocketHandler
        A socket handler configured to send log messages to the specified host and port.

    Notes:
    ------
    - Assumes a global `FORMATTER` object is defined to format log messages.
    - Defaults to "localhost" and port 9999 if no host or port is specified.
    - Can be used to send log messages to a centralized logging server or custom application.
    """
    socket_handler = handlers.SocketHandler(host=host, port=port)
    socket_handler.setFormatter(FORMATTER)
    return socket_handler


def get_file_logger(logger_name: str, path: str) -> logging.Logger:
    """
    Creates and returns a file-based logger.

    This function initializes a logger with the specified name and configures it to write
    log messages to a file. The logger captures all log levels (DEBUG and above) and uses
    a file handler for managing log files.

    Parameters:
    ----------
    logger_name : str
        The name of the logger, typically used to identify the logger in the application.
    path : str
        The full file path where the log file will be stored. Parent directories are created
        if they do not already exist.

    Returns:
    -------
    logging.Logger
        A configured logger instance.

    Notes:
    ------
    - The `get_file_handler` function must be defined and available to create a file handler.
    - Sets the logger's logging level to DEBUG to capture all log levels.
    - Disables propagation to avoid duplicate log entries in parent loggers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler(path))
    logger.propagate = False
    return logger


def get_console_logger(logger_name: str) -> logging.Logger:
    """
    Creates and returns a console-based logger.

    This function initializes a logger with the specified name and configures it to output
    log messages to the console. The logger captures all log levels (DEBUG and above) and
    uses a console handler.

    Parameters:
    ----------
    logger_name : str
        The name of the logger, typically used to identify the logger in the application.

    Returns:
    -------
    logging.Logger
        A configured logger instance.

    Notes:
    ------
    - The `get_console_handler` function must be defined and available to create a console handler.
    - Sets the logger's logging level to DEBUG to capture all log levels.
    - Disables propagation to avoid duplicate log entries in parent loggers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.propagate = False
    return logger


def get_basic_logger(
    logger_name: str,
) -> logging.Logger:
    """
    Creates and returns a basic logger.

    This function initializes a logger with the specified name and sets its logging level to DEBUG.
    It does not add any handlers, leaving the configuration of handlers to the user.

    Parameters:
    ----------
    logger_name : str
        The name of the logger, typically used to identify the logger in the application.

    Returns:
    -------
    logging.Logger
        A configured logger instance.

    Notes:
    ------
    - The logger's level is set to DEBUG to capture all log levels.
    - Propagation is disabled to prevent duplicate log entries in parent loggers.
    - Handlers are not added; the user must configure handlers as needed.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
