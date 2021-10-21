import pytest
import shutil
import pytest
from learning_loop_node.globals import GLOBALS
from learning_loop_node.trainer.training_data import TrainingData
import yolo_helper
import yolo_cfg_helper
import os
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
import helper
from tests import test_helper
from learning_loop_node.conftest import create_project


@pytest.mark.asyncio
async def test_yolo_box_creation(create_project):
    darknet_trainer = test_helper.create_darknet_trainer()
    await test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data

    # 3 images, 3 model files
    assert len(test_helper.get_files_from_data_folder()) == 6

    image_folder_for_training = yolo_helper.create_image_links(
        f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid', f'{GLOBALS.data_folder}/zauberzeug/pytest/images', training_data.image_ids())

    await yolo_helper.update_yolo_boxes(image_folder_for_training, training_data)

    # # 3 images, 3 model files,   3 image_links, 3 txt files
    assert len(test_helper.get_files_from_data_folder()) == 12

    first_image_id = training_data.image_ids()[0]
    with open(f'{image_folder_for_training}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()

    assert yolo_content == '''0 0.725000 0.721250 0.050000 0.057500
2 0.075000 0.201250 0.050000 0.057500
2 0.350000 0.317083 0.050000 0.057500'''


def test_create_names_file(data_folder):
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, training_folder = trainer_test_helper.create_needed_folders(GLOBALS.data_folder)

    yolo_helper.create_names_file(training_folder, ['category_1', 'category_2'])
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    assert files[0].endswith('names.txt')

    with open(f'{training_folder}/names.txt', 'r') as f:
        names = f.readlines()

    assert len(names) == 2
    assert names[0] == 'category_1\n'
    assert names[1] == 'category_2'


def test_create_data_file(data_folder):
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, training_folder = trainer_test_helper.create_needed_folders(GLOBALS.data_folder)

    yolo_helper.create_data_file(training_folder, 1)
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    data_file = files[0]
    assert data_file.endswith('data.txt')
    with open(f'{training_folder}/data.txt', 'r') as f:
        data = f.readlines()

    assert len(data) == 5
    assert data[0] == 'classes = 1\n'
    assert data[1] == 'train  = train.txt\n'
    assert data[2] == 'valid  = test.txt\n'
    assert data[3] == 'names = names.txt\n'
    assert data[4] == 'backup = backup/'


@pytest.mark.asyncio
async def test_create_image_links(create_project):
    assert len(test_helper.get_files_from_data_folder()) == 0

    darknet_trainer = test_helper.create_darknet_trainer()
    await test_helper.downlaod_data(darknet_trainer)
    # 3 images, 3 model files
    assert len(test_helper.get_files_from_data_folder()) == 6

    training = darknet_trainer.training
    training_data = training.data
    training_id = training.id

    _ = yolo_helper.create_image_links(f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid',
                                       f'{GLOBALS.data_folder}/zauberzeug/pytest/images', training_data.image_ids())
    files = test_helper.get_files_from_data_folder()

    # 3 images, 3 model files,   3 image links
    assert len(files) == 9

    assert f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/285a92db-bc64-240d-50c2-3212d3973566.jpg' in files
    assert f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/6a4ddab1-93b4-b2e2-30c5-16b58f46d0d0.jpg' in files
    assert f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/a120df7c-51ec-b22e-d012-c9ee745fcc8e.jpg' in files


@pytest.mark.asyncio
async def test_create_train_and_test_file(create_project):
    assert len(test_helper.get_files_from_data_folder()) == 0
    darknet_trainer = test_helper.create_darknet_trainer()
    await test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    print(training.training_folder)
    print(training.images_folder)
    training_data = training.data

    images_folder_for_training = yolo_helper.create_image_links(
        f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid', f'{GLOBALS.data_folder}/zauberzeug/pytest/images', training_data.image_ids())

    yolo_helper.create_train_and_test_file(
        f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid', images_folder_for_training, training_data.image_data)

    files = [file for file in test_helper.get_files_from_data_folder() if file.endswith('test.txt')
             or file.endswith('train.txt')]
    print(files)
    assert len(files) == 2
    test_file = files[0]
    train_file = files[1]
    assert train_file.endswith('train.txt')
    assert test_file.endswith('test.txt')
    with open(f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/train.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 2
    assert content[0] == f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/6a4ddab1-93b4-b2e2-30c5-16b58f46d0d0.jpg\n'
    assert content[1] == f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/285a92db-bc64-240d-50c2-3212d3973566.jpg\n'

    with open(f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/test.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 1
    assert content[0] == f'{GLOBALS.data_folder}/zauberzeug/pytest/trainings/some_model_uuid/images/a120df7c-51ec-b22e-d012-c9ee745fcc8e.jpg\n'


def test_replace_classes_and_filters():
    target_folder = '/tmp/classes_test'

    def get_yolo_lines():
        with open(f'{target_folder}/training.cfg', 'r') as f:
            return f.readlines()

    def assert_line_count(search_string, expected_count):
        matched_lines = [line for line in get_yolo_lines() if line.strip() == search_string]
        assert len(matched_lines) == expected_count

    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder)

    shutil.copy('tests/integration/data/training.cfg', f'{target_folder}/training.cfg')

    assert_line_count('filters=45', 0)
    assert_line_count('classes=10', 0)

    yolo_cfg_helper.replace_classes_and_filters(10, target_folder)

    assert_line_count('filters=45', 2)
    assert_line_count('classes=10', 2)


@pytest.mark.asyncio
async def test_create_anchors(create_project):
    assert len(test_helper.get_files_from_data_folder()) == 0

    darknet_trainer = test_helper.create_darknet_trainer()
    await test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data

    image_folder_for_training = yolo_helper.create_image_links(
        training.training_folder, training.images_folder, training_data.image_ids())
    await yolo_helper.update_yolo_boxes(image_folder_for_training, training_data)
    box_category_names = helper.get_box_category_names(training_data)
    yolo_helper.create_names_file(training.training_folder, box_category_names)
    yolo_helper.create_data_file(training.training_folder, len(box_category_names))
    yolo_helper.create_train_and_test_file(
        training. training_folder, image_folder_for_training, training_data.image_data)
    yolo_cfg_helper.update_anchors(training.training_folder)

    anchor_line = 'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319'
    original_cfg_file_path = yolo_cfg_helper._find_cfg_file('tests/integration/data')
    _assert_anchors(original_cfg_file_path, anchor_line)

    new_anchors = 'anchors=1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400'
    cfg_file_path = yolo_cfg_helper._find_cfg_file(training.training_folder)
    _assert_anchors(cfg_file_path, new_anchors)


def test_find_cfg_file():
    _, _, training_path = trainer_test_helper.create_needed_folders(GLOBALS.data_folder)

    shutil.copy(f'tests/integration/data/training.cfg', f'{training_path}/training.cfg')
    found_cfg_file = yolo_cfg_helper._find_cfg_file(training_path)
    assert 'training.cfg' in found_cfg_file


def test_convert_points_into_small_boxes():

    traings_data = TrainingData(image_data=[
        {
            'point_annotations': [{'x': 100, 'y': 100, 'category_id': 42}],
            'box_annotations': []
        }],
        categories=[])

    yolo_helper.convert_points_into_small_boxes(traings_data)
    boxes = traings_data.image_data[0].get('box_annotations', [])
    assert len(boxes) == 1

    assert boxes[0] == {'x': 90, 'y': 90, 'width': 20, 'height': 20, 'category_id': 42}


def _assert_anchors(cfg_file_path: str, anchor_line: str) -> None:
    anchor_line = anchor_line.replace(' ', '')
    found_anchor_line_count = 0
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace(' ', '').strip()
        if line.startswith('anchors='):
            assert line == anchor_line, 'Anchor line does not match. '
            found_anchor_line_count += 1
    assert found_anchor_line_count > 0, 'There must be at least one anchorline in cfg file.'
