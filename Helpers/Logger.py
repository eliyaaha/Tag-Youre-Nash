import logging
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, experiment_name, alpha):
        self.alpha = alpha
        # Create the logs folder if doesn't exist
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{self.log_dir}/{experiment_name}_alpha_{alpha}_{timestamp}.log"
        
        # Set the log's format
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger(f"{experiment_name}_{alpha}")
        self.logger.setLevel(logging.INFO)

        # write to file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        # write to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def log_metrics(self, ep, reward, caps, collisions):
        self.logger.info(f"EP {ep} | Reward: {reward:.2f} | Caps: {caps} | Collisions: {collisions}")