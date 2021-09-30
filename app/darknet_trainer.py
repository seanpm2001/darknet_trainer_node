import shutil
from typing import Any, List, Optional, Union
from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.trainer.trainer import Trainer
import yolo_helper
import helper
import yolo_cfg_helper
import os
import model_updater
import logging


class DarknetTrainer(Trainer):

    def __init__(self, model_format: str) -> None:
        super().__init__(model_format)
        self.latest_published_iteration: Union[int, None] = None

    async def start_training(self) -> None:
        await self.prepare_training()
        training_path = self.training.training_folder
        weightfile = yolo_helper.find_weightfile(training_path)
        cfg_file = yolo_cfg_helper._find_cfg_file(training_path)

        self.executor.start(f'/darknet/darknet detector train data.txt {cfg_file} {weightfile} -dont_show -map -clear')

    async def prepare_training(self) -> None:
        training_folder = self.training.training_folder
        image_folder = self.training.images_folder
        training_data = self.training.data
        yolo_helper.create_backup_dir(training_folder)
        with open(training_folder + '/training.cfg', 'r') as f:
            logging.info('before anything')
            logging.info(f.read(1000))

        image_folder_for_training = yolo_helper.create_image_links(
            training_folder, image_folder, training_data.image_ids())
        await yolo_helper.update_yolo_boxes(image_folder_for_training, training_data)
        box_category_names = helper.get_box_category_names(training_data)
        yolo_helper.create_names_file(training_folder, box_category_names)
        yolo_helper.create_data_file(training_folder, len(box_category_names))
        yolo_helper.create_train_and_test_file(
            training_folder, image_folder_for_training, training_data.image_data)
        with open(training_folder + '/training.cfg', 'r') as f:
            logging.info('before cfg helper')
            logging.info(f.read(1000))

        yolo_cfg_helper.replace_classes_and_filters(len(box_category_names), training_folder)
        yolo_cfg_helper.update_anchors(training_folder)
        yolo_cfg_helper.update_hyperparameters(training_folder)

    def get_error(self) -> str:
        if self.executor is None:
            return
        try:
            if 'CUDA Error: out of memory' in self.executor.get_log():
                return 'graphics card is out of memory'
        except:
            return

    def get_model_files(self, model_id) -> List[str]:
        from glob import glob
        try:
            weightfile_path = glob(f'/data/**/trainings/**/{model_id}.weights', recursive=True)[0]
            os.makedirs(f'/tmp/{model_id}', exist_ok=True)
            os.makedirs(weightfile_path.replace('.weights', ''), exist_ok=True)
            model_dot_weights_path = f'/tmp/{model_id}/model.weights'
            shutil.copy(weightfile_path, model_dot_weights_path)
        except:
            raise Exception(f'No model found for id: {model_id}.')
        training_path = '/'.join(weightfile_path.split('/')[:-1])  # hier
        cfg_file_path = yolo_cfg_helper._find_cfg_file(training_path)
        return [model_dot_weights_path, f'{cfg_file_path}', f'{training_path}/names.txt']

    def get_new_model(self) -> Optional[BasicModel]:
        return model_updater.check_state(self.training.id, self.training.data, self.latest_published_iteration)

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        self.latest_published_iteration = basic_model.meta_information['iteration']
        weightfile_path = basic_model.meta_information['weightfile_path']
        path = weightfile_path.rsplit('/', 2)[0]
        new_filename = path + f'/{uuid}.weights'
        shutil.move(weightfile_path, new_filename)

    def stop_training(self) -> None:
        self.executor.stop()

    def _show_log(self) -> str:
        if not self.training:
            raise Exception('no training running')
        return self.executor.get_log()
