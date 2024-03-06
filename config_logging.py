import os
import logging
import logging.config
import yaml

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message.strip())

    def flush(self):
        pass

def setup_logging(config_path):
    if(not os.path.exists("logs")):
        os.mkdir("logs")   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    import sys
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.warning)


setup_logging('logging_config.yaml')