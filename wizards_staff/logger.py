import logging

# logger_config.py
def init_custom_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a custom logger with the specified name.
    Args:
        name: The name of the logger.
    Returns:
        The custom logger.
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the desired logging level

    # Check if the logger already has handlers to avoid duplicate handlers
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(message)s')  # Adjust the format as needed
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Prevent log messages from propagating to the root logger
        logger.propagate = False

    return logger