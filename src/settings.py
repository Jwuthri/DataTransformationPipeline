import logging

from rich.logging import RichHandler


# DEBUG
DEBUG = False

# LOG
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
LOGGER = logging.getLogger("rich")
