import asyncio
from threading import Thread
from learning_loop_node import TrainerNode
from darknet_trainer import DarknetTrainer
import uvicorn
import logging
import os

logging.basicConfig(level=logging.INFO)

darknet_trainer = DarknetTrainer(model_format='yolo')
node = TrainerNode(
    name='darknet trainer ' + os.uname()[1],
    trainer=darknet_trainer
)


@node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio_client.disconnect())
        darknet_trainer.stop_training()

    Thread(target=restart).start()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
