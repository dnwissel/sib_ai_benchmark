import logging

class Logger:
    def __init__(self, name='App', log_to_file=True, log_to_console=True):
        self.logger = self.__get_logger(log_to_file, log_to_console, name)

    def write(self, msg, msg_type):
        msg_types = ['title', 'subtitle', 'content']
        if msg_type not in msg_types:
            raise ValueError(f"Invalid msg_type '{msg_type}'. Valid values are {', '.join(msg_types)}.")

        if msg_type == 'title':
            formatter = logging.Formatter('[%(asctime)s]  %(name)s - %(levelname)s - %(message)s')
        elif msg_type == 'subtitle':
            formatter = logging.Formatter('[%(asctime)s]  %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')

        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

        self.logger.info(msg)

    def __get_logger(self, log_to_file, log_to_console, name):
        if not log_to_console and not log_to_file:
            raise ValueError('At least one of log_to_file and log_to_console must be true.')

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            logger.addHandler(ch)

        if log_to_file:
            fh = logging.FileHandler(f'{name}.log')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)

        return logger


if __name__ == "__main__":
    logger = Logger()
    logger.write('This is a title', 'title')
    logger.write('This is a subtitle', 'subtitle')
    logger.write('This is a content message', 'content')

