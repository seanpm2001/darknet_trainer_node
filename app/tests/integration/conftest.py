import pytest
from typing import Generator
from learning_loop_node.tests import test_helper


@pytest.fixture()
def web() -> Generator:
    with test_helper.LiveServerSession() as c:
        yield c
