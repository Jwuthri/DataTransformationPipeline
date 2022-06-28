import logging

from rich.logging import RichHandler


# DEBUG
DEBUG = True

# LOG
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
LOGGER = logging.getLogger("rich")
