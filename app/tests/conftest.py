import icecream
import logging
from learning_loop_node import loop
import pytest
from learning_loop_node.globals import GLOBALS
import shutil

icecream.install()
logging.basicConfig(level=logging.INFO)

loop.base_url = 'https://preview.learning-loop.ai'


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
