from learning_loop_node.context import Context
import pytest
from typing import Generator
import learning_loop_node.tests.test_helper as test_helper
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
import tests.test_helper as darknet_test_helper
import shutil
import os
import asyncio
from icecream import ic


@pytest.fixture()
def web() -> Generator:
    with test_helper.LiveServerSession() as c:
        yield c


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'empty',
                             'box_categories': 2, 'point_categories': 1, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'box_detections': 3, 'box_annotations': 1, 'point_annotations': 1}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.mark.asyncio
async def test_start_stop_training():
    model_id = await trainer_test_helper.assert_upload_model(
        ['tests/integration/data/training.cfg', 'tests/integration/data/model.weights'],
        'yolo'
    )

    darknet_trainer = darknet_test_helper.create_darknet_trainer()

    assert darknet_trainer.get_error() is None
    context = Context(organization='zauberzeug', project='pytest')
    await darknet_trainer.begin_training(context=context, source_model={'id': model_id})
    await asyncio.sleep(1)
    assert darknet_trainer.get_error() is None
    assert 'CUDA-version' in darknet_trainer.get_log()

    darknet_trainer.stop_training()
    assert darknet_trainer.get_error() is None


@pytest.mark.asyncio
async def test_get_model_files():
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)
    shutil.copy('tests/integration/data/model.weights',
                f'{darknet_trainer.training.training_folder}/some_model_uuid.weights')
    files = darknet_trainer.get_model_files('some_model_uuid')

    assert len(files) == 3
    assert files[0].endswith('/model.weights')
    assert files[1].endswith('/training.cfg')
    assert files[2].endswith('/names.txt')
    for f in files:
        assert os.path.exists(f)


@pytest.mark.asyncio
async def test_get_new_model():
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)

    path = f'{darknet_trainer.training.training_folder}/backup/'
    os.makedirs(path)
    open(f'{path}/tiny_yolo_best_mAP_0.000000_iteration_1088_avgloss_-nan_.weights', 'a').close()

    shutil.copy('tests/integration/data/last_training.log',
                f'{darknet_trainer.training.training_folder}/last_training.log')

    model = darknet_trainer.get_new_model()
    assert model is not None

    # test: store weightfile for further use.
    with pytest.raises(Exception):
        files = darknet_trainer.get_model_files('some_model_uuid')

    darknet_trainer.on_model_published(model, 'some_model_uuid')
    files = darknet_trainer.get_model_files('some_model_uuid')
    assert len(files) == 3
    model_dot_weights_files = [path for path in files if 'model.weights' in path]
    assert len(model_dot_weights_files) == 1, "There must be a file named model.weights"
    assert os.path.isfile(model_dot_weights_files[0])


@pytest.mark.asyncio
async def test_point_is_added_when_training_starts():
    model_id = await trainer_test_helper.assert_upload_model(
        ['tests/integration/data/training.cfg', 'tests/integration/data/model.weights'],
        'yolo'
    )
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    context = Context(organization='zauberzeug', project='pytest')

    await darknet_trainer.begin_training(context=context, source_model={'id': model_id})
    await asyncio.sleep(1)
    assert darknet_trainer.get_error() is None
    assert 'CUDA-version' in darknet_trainer.get_log()

    training_data = darknet_trainer.training.data
    assert len(training_data.image_data) == 3
    print(*training_data.categories, sep='\n')

    assert len(training_data.categories) == 3, 'There should be 3 categories, 2 boxes and 1 point'

    point_categories = [c for c in training_data.categories if c['type']
                        == 'point']
    assert len(point_categories) == 1, 'There should be one point category'
    point_category = point_categories[0]

    assert len(training_data.image_data[0]['point_annotations']) == 1
    assert len(training_data.image_data[0]['box_annotations']
               ) == 2, 'There should be two box_annotations. One original and one small_box converted from point'

    assert training_data.image_data[0]['box_annotations'][1]['category_id'] == point_category['id']

    darknet_trainer.stop_training()

