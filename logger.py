import os
import logging
from cloghandler import ConcurrentRotatingFileHandler
log_format = '[Time:%(asctime)s - Filename:%(filename)s - Line:%(lineno)d - PID:%(process)d - Level:%(levelname)s] : %(message)s'
filedir = os.path.dirname(__file__)
class Logger:
    def __init__(self,path,logname,max_bytes,backup_count):
        self.logpath = path
        self.logname = logname
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)

    def create_logger(self):
        level = logging.INFO
        formatter = log_format
        logger = logging.getLogger(__name__)
        mode = 'a'
        delay = 0
        encoding = 'utf-8'
        if not os.path.exists(self.logpath):
            os.mkdir(self.logpath)
        log_filename = os.path.join(self.logpath, self.logname)
        handler = ConcurrentRotatingFileHandler(log_filename,
                                                mode=mode,
                                                maxBytes=self.max_bytes,
                                                backupCount=self.backup_count,
                                                delay=delay,
                                                encoding=encoding)
        handler.setFormatter(logging.Formatter(formatter))
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

