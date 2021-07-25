import icecream
import logging
from learning_loop_node import loop

icecream.install()
logging.basicConfig(level=logging.DEBUG)

loop.base_url = 'https://preview.learning-loop.ai'
