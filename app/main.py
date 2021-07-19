import asyncio
from threading import Thread
from learning_loop_node import TrainerNode
from learning_loop_node import Capability
from darknet_trainer import DarknetTrainer
import uvicorn
import logging

logging.basicConfig(level=logging.DEBUG)

darknet_trainer = DarknetTrainer(capability=Capability.Box, model_format='yolo')
node = TrainerNode(uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4', name='darknet trainer', trainer=darknet_trainer)


@node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio_client.disconnect())

    Thread(target=restart).start()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
