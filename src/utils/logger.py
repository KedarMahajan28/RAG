import sys
import logging

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured with standard output and formatting.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger
