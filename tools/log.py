import os
import sys
import logging
import datetime

def create_logger(file_name = None) -> logging.Logger:
    """
    Retrieve an existing logger or create a new one if none exists.
    
    If the logger
    doesn't have any handlers, it configures logging to write to a file named
    'log.log' in the 'log' directory, sets the logging level to INFO, and
    formats log messages to include the timestamp, log level, and message.
    Additionally, it configures a console handler to write log messages to the
    console with the same format.

    Parameters
    ----------
    file_name : str
        File name to write log messages

    Returns:
    --------
    logging.Logger
        The shared logger instance.
    """

    logger = logging.getLogger()

    # Configure logging if no handler exists
    if file_name is not None:
        # Make dir
        out_dir = 'log'
        os.makedirs(out_dir, exist_ok=True)
        # Cretae file path
        file_path = os.path.join(out_dir, file_name)

        logging.basicConfig(
            filename=file_path,
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(message)s'
        )

        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
                )
            )
        logger.addHandler(console_handler)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(message)s'
        )

    return logger

logger = create_logger('log.log')
