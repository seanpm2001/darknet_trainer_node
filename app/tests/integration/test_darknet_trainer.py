from learning_loop_node.context import Context
import pytest
from typing import Generator
from learning_loop_node.globals import GLOBALS
import learning_loop_node.tests.test_helper as test_helper
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.training_data import TrainingData
import tests.test_helper as darknet_test_helper
import shutil
import os
import asyncio
from icecream import ic
from tests import test_helper as darknet_test_helper


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
async def test_points_creation():
    training_data = TrainingData(image_data=[
        {
            'id': 'some_image_id',
            'point_annotations': [{'x': 100, 'y': 100, 'category_id': 'point_1_id'}],
            'box_annotations': [],

            'width': 1000,
            'height': 1000,
            'set': 'train'

        }],
        categories=[{'id': 'point_1_id', 'name': 'point_1', 'type': 'point'}]
    )
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    darknet_trainer.training = create_training(training_data)

    await darknet_trainer.prepare_training()

    with open(f'{darknet_trainer.training.training_folder}/images/some_image_id.txt', 'r') as f:
        content = f.read()
        assert content == '0 0.100000 0.100000 0.020000 0.020000'


@pytest.mark.asyncio
async def test_box_creation():
    training_data = TrainingData(image_data=[
        {
            'id': 'some_image_id',
            'point_annotations': [],
            'box_annotations': [{'x': 90, 'y': 90, 'width': 20, 'height': 20, 'category_id': 'box_1_id'}],

            'width': 1000,
            'height': 1000,
            'set': 'train'

        }],
        categories=[{'id': 'box_1_id', 'name': 'box_1', 'type': 'box'}]
    )

    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    darknet_trainer.training = create_training(training_data)

    await darknet_trainer.prepare_training()

    with open(f'{darknet_trainer.training.training_folder}/images/some_image_id.txt', 'r') as f:
        content = f.read()
        assert content == '0 0.100000 0.100000 0.020000 0.020000'


def create_training(training_data: TrainingData) -> Training:
    data_folder = GLOBALS.data_folder

    images_folder = f"{data_folder}/images"
    os.makedirs(images_folder)
    image_path = f'{images_folder}/project_folder.jpg'
    open(image_path, 'a').close()

    project_folder = f'{data_folder}/some_project'
    os.makedirs(project_folder)
    training_folder = f'{project_folder}/some_training_id'
    os.makedirs(training_folder)

    training = Training(id='some_training_id', context=Context(
        project='p', organization='o'), images_folder=images_folder, training_folder=training_folder, project_folder=project_folder, data=training_data)

    shutil.copy('tests/integration/data/training.cfg',
                f'{training.training_folder}/training.cfg')
    return training


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
