import json

import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import fl_config_from_json
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

from script.ResultToCSV import CreateResultData, Save_Accuracy_of_each_epoch
from script.getKL import saveKL
from src.model.MLP import MNIST_MLP
from src.sampling import mnist_noniid

IMAGE_SIZE = 28


def build_data_provider(local_batch_size, examples_per_user, drop_last: bool = False):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="../Experiment/data/MNIST", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="../Experiment/data/MNIST", train=False, download=True, transform=transform
    )
    client_num = int(len(train_dataset) / examples_per_user)
    dict_users = mnist_noniid(train_dataset, client_num)
    sorted_train_dataset = []

    # merge train dataset
    for k in range(client_num):
        for i in range(examples_per_user):
            index = int(dict_users[k][i])
            sorted_train_dataset.append(train_dataset[index])

    saveKL(train_dataset=sorted_train_dataset,
           label=10,
           num_of_client=client_num,
           fileName="FL_non_IID_MNIST(MLP)"
           )
    # get the amount of each class
    num_of_class_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(sorted_train_dataset)):
        index = sorted_train_dataset[i][1]
        num_of_class_list[index] = num_of_class_list[index] + 1
    # print(num_of_class_list)

    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoader(
        sorted_train_dataset,
        test_dataset,
        test_dataset,
        sharder,
        local_batch_size,
        drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def main(
        trainer_config,
        data_config,
        use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    model = MNIST_MLP()
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    assert (global_model.fl_get_module() == model)

    if cuda_enabled:
        global_model.fl_cuda()
    # print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    metrics_reporter: MetricsReporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )
    accuracy_of_each_epoch = metrics_reporter.AccuracyList
    best_accuracy_of_each_epoch = max(accuracy_of_each_epoch)

    Save_Accuracy_of_each_epoch(1, "FL_non_IID_MNIST(MLP)", accuracy_of_each_epoch, best_accuracy_of_each_epoch)
    client_num = data_provider.num_train_users()
    CreateResultData("FL_non_IID_MNIST(MLP)", "MNIST", "MLP", "non-IID", client_num, int(trainer_config.epochs),
                     eval_score['Accuracy'], "")


@hydra.main(config_path="configs", config_name="MNIST_config", version_base="1.2")
def run(cfg: DictConfig) -> None:
    print('-------------------FL_non_IID_MNIST(MLP)-------------------')
    # print(cfg)
    trainer_config = cfg.trainer
    data_config = cfg.data
    main(
        trainer_config,
        data_config
    )


if __name__ == "__main__":
    cfg = OmegaConf.create(fl_config_from_json(json.load(open('./configs/MNIST_config.json'))))
    run(cfg)