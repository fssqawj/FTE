import logging
class Logger:
    def __init__(self, logName='RESCAL', logFile='./log/rescal.log', logFormat="%(asctime)s:%(name)s %(message)s", logLevel=logging.INFO):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(logFormat))
        logging.basicConfig(filename=logFile, filemode='w', level=logLevel, format=logFormat)
        self._logger = logging.getLogger(logName) 
        self._logger.addHandler(ch)

    def getLog(self):
        return self._logger

    def setLog(self, log):
        self._logger = log

    def setLogLevel(self, level):
        self._logger.setLevel(level)

    def setLogFormat(self, format):
        self.setFormatter(format)
