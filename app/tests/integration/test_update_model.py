from learning_loop_node.trainer.downloader import DataDownloader as Downloader
from learning_loop_node.context import Context
from learning_loop_node.trainer.trainer import Trainer
import shutil
import pytest
import test_helper
import model_updater
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
from learning_loop_node.conftest import create_project


@pytest.fixture
def downloader() -> Downloader:
    return test_helper.create_downloader()


@pytest.mark.asyncio
async def test_parse_latest_confusion_matrix(downloader: Downloader, create_project):
    model_id = await trainer_test_helper.assert_upload_model(
        ['tests/integration/data/training.cfg', 'tests/integration/data/model.weights'])
    context = Context(organization='zauberzeug', project='pytest')
    training = Trainer.generate_training(context, {'id': 'some_uuid'})
    training.data = await downloader.download_data(training.images_folder)

    shutil.copy('tests/integration/data/last_training.log', f'{training.training_folder}/last_training.log')

    new_model = model_updater._parse_latest_iteration(training.id, training.data)
    assert new_model
    assert new_model['iteration'] == 1088
    confusion_matrix = new_model['confusion_matrix']
    assert len(confusion_matrix) == 2
    purple_matrix = confusion_matrix[training.data.categories[0]['id']]

    assert purple_matrix['ap'] == 42
    assert purple_matrix['tp'] == 1
    assert purple_matrix['fp'] == 2
    assert purple_matrix['fn'] == 3

    weightfile = new_model['weightfile']
    assert weightfile == 'backup/tiny_yolo_best_mAP_0.000000_iteration_1088_avgloss_-nan_.weights'


def get_box_categories():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/data').json()
    categories = content['box_categories']
    return categories


def get_model_ids_from__latest_training():
    content = test_helper.LiveServerSession().get('/api/zauberzeug/projects/pytest/trainings')
    content_json = content.json()
    datapoints = [datapoint['model_id'] for datapoint in content_json['charts'][0]['data']]
    return datapoints
