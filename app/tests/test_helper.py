from learning_loop_node.trainer.downloader import DataDownloader as Downloader
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.context import Context
from learning_loop_node.trainer.trainer import Trainer
from darknet_trainer import DarknetTrainer
from glob import glob
import os
from learning_loop_node.trainer.capability import Capability
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
from learning_loop_node import node
from learning_loop_node import node_helper
from learning_loop_node import conftest


def get_files_from_data_folder():
    files = [entry for entry in glob(f'{conftest.data_folder_for_tests}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


def create_darknet_trainer() -> DarknetTrainer:
    return DarknetTrainer(model_format='yolo', capability=Capability.Box)


def create_downloader() -> Downloader:
    context = Context(organization='zauberzeug', project='pytest')
    return DownloaderFactory.create(context=context, capability=Capability.Box)


async def downlaod_data(trainer: Trainer):
    model_id = await trainer_test_helper.assert_upload_model(
        ['tests/integration/data/' + f for f in ['fake_weightfile.weights', 'training.cfg', 'names.txt']],
        format='yolo'
    )
    context = Context(organization='zauberzeug', project='pytest')
    training = Trainer.generate_training(context, {'id': model_id})
    downloader = create_downloader()
    training.data = await downloader.download_data(training.images_folder)
    await node_helper.download_model(training.training_folder, context, model_id, 'yolo')

    trainer.training = training
