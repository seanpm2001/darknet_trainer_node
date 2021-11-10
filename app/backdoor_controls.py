
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.error_configuration import ErrorConfiguration
from fastapi import APIRouter,  Request,  HTTPException
from learning_loop_node.status import Status, State
import logging
from icecream import ic

router = APIRouter()


@router.get("/status")
async def status(request: Request):
    trainer_node = trainer_node_from_request(request)
    status = {}
    status['sio'] = {
        'connected': trainer_node.sio_client.connected
    }
    if trainer_node.trainer.training:
        status['training'] = trainer_node.trainer.training.__dict__.copy()
        # print(status)
        status['training']['data'] = ""

    # status['excecutor'] = trainer_node.trainer.executor.__dict__ if trainer_node.trainer and trainer_node.trainer.excecutor else None
    #ic(trainer_node.trainer.executor)
    return status


@router.get("/kill_training")
async def kill_training(request: Request):
    trainer_node = trainer_node_from_request(request)
    trainer = trainer_node.trainer
    if trainer.executor:
        pid = trainer.executor.process.pid
        trainer.executor.stop()


def trainer_node_from_request(request: Request) -> TrainerNode:
    return request.app
