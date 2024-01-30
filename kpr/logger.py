import os
import logging
from datetime import datetime 

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs",LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(level = logging.INFO, 
                    filename = LOG_FILE_PATH, 
                    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    logging.info("logger python file working properly")