import logging
import sys

def setup_logging():

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    
    #configure root logger
    logging.basicConfig(
        level=logging.INFO, 
        format=log_format, 
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    #silence overly chatty libraries 
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

