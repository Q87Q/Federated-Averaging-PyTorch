import datetime
import logging
import os
import pickle
import threading
import time

import yaml
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
from src.server import Server
from src.utils import launch_tensor_board


def readConfigFile(filePath):
    # read configuration file
    with open(filePath) as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"],
                                          str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    for config in configs:
        print(config)
        logging.info(config)
    print()
    return global_config, data_config, fed_config, optim_config, init_config, model_config, log_config


if __name__ == "__main__":
    _, _, fed_config, _, _, _, log_config = readConfigFile("./config.yaml")

    writer = SummaryWriter(log_dir=log_config["log_path"],
                           filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"],
               log_config["tb_port"],
               log_config["tb_host"]])
    ).start()
    time.sleep(3.0)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"],
                              log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message)
    logging.info(message)

    configFileList = [f for f in listdir('./configs') if isfile(join('./configs', f))]
    for file in configFileList:
        global_config, data_config, fed_config, optim_config, init_config, model_config, log_config = readConfigFile("./configs/"+file)
        # initialize federated learning
        central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
        central_server.setup()
        # do federated learning
        central_server.fit()
        # save resulting losses and metrics
        with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
            pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
    logging.info(message)
    time.sleep(3)
    exit()
