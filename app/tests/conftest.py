import icecream
import logging
from learning_loop_node import loop
import pytest
from learning_loop_node.globals import GLOBALS
import shutil
import asyncio

icecream.install()
logging.basicConfig(level=logging.INFO)

loop.base_url = 'https://preview.learning-loop.ai'


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)


@pytest.fixture(scope="session")
def event_loop(request):
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
