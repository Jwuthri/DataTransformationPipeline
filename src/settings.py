import os
import logging

from rich.logging import RichHandler


# DEBUG
DEBUG = True

# LOG
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
LOGGER = logging.getLogger("rich")

# PATH
ABSOLUTE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ABSOLUTE_PATH, "data")
FIXTURES_PATH = os.path.join(ABSOLUTE_PATH, "src", "fixtures")

PROCESSED_DATA = os.path.join(DATA_PATH, "processed")
EXTERNAL_DATA = os.path.join(DATA_PATH, "external")
INTERIM_DATA = os.path.join(DATA_PATH, "interim")
RAW_DATA = os.path.join(DATA_PATH, "raw")
