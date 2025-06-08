"""
This file helps to create a log message and store it in a log directory
which can be used to track the execution of the script. This helps
to identify what is the error and where did it occur
"""
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)
"""
A basic configuration for this log message has been created 
with the format of the log message we want.
"""

if __name__ == "__main__":
    logging.info("Logging has strated")