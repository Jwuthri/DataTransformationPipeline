import time

from src.settings import LOGGER


def timeit(method):
    """
    :param method:
    :return:
    """
    def wrapper(*args, **kw):
        """
        :param args:
        :param kw:
        :return:
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        msg = '%r duration %2.2f ms' % (method.__name__, (te - ts) * 1000)
        LOGGER.info(msg)

        return result

    return wrapper