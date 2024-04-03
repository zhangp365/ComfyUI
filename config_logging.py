import os
import logging
import logging.config
import yaml
import signal
import re

watch_log_keyword_kill_process = False


app_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(app_dir,"config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.encoding = None

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




class KeywordWatcher(logging.Handler):
    def __init__(self, regex):
        super().__init__()
        self.regex = re.compile(regex)

    def emit(self, record):
        log_message = self.format(record)
        if self.regex.search(log_message):
            print(f"Detected keyword in log, start to kill the current process!!!")
            os.kill(os.getpid(), signal.SIGINT)

# 加载 YAML 文件中的关键字
def load_keywords_from_yaml(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write("RuntimeError: CUDA error")
    with open(filename, 'r') as file:
        keywords = file.readlines()
    return '|'.join(keywords)



setup_logging('logging_config.yaml')

if watch_log_keyword_kill_process:
    # 从 YAML 文件中加载关键字并构建正则表达式
    keywords_regex = load_keywords_from_yaml(os.path.join(config_dir,'sys_exit_with_keywords_in_log.yaml'))
    root_logger = logging.getLogger()
    root_logger.addHandler(KeywordWatcher(keywords_regex))

