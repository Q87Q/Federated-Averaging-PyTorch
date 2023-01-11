import json

import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import fl_config_from_json
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from algorithm.Heuristic_Algorithm import heuristic_method
from configs.ILP_Heuristic_method_parameter import (
    num_of_original_client,
    num_of_head_client,
    num_of_CIFAR10_label,
)
from model.CIFAR10_CNN import CIFAR10_CNN
from script.ResultToCSV import CreateResultData, Save_KL_Result, Save_Accuracy_of_each_epoch
from script.getKL import get_KL_value
from script.non_iid import cifar10_noniid

IMAGE_SIZE = 32
total_execution_time = 0


def build_data_provider(local_batch_size, examples_per_user, drop_last: bool = False):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="./data/Experiment/data/cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="./data/Experiment/data/cifar10", train=False, download=True, transform=transform
    )

    client_num = num_of_original_client

    dict_users = cifar10_noniid(train_dataset, client_num)
    index_of_head_group, time = heuristic_method(train_dataset, num_of_CIFAR10_label, dict_users, client_num,
                                                 num_of_head_client)
    global total_execution_time
    total_execution_time = total_execution_time + time

    # merge train dataset
    sorted_train_dataset = []
    # print(len(dict_users[0]))
    for k in range(num_of_head_client):

        for i in range(len(index_of_head_group[k])):
            client_index = index_of_head_group[k][i]
            for j in range(len(dict_users[client_index])):
                data_index = int(dict_users[client_index][j])
                sorted_train_dataset.append(train_dataset[data_index])

    num_of_client = int(len(train_dataset) / examples_per_user)

    KL_of_each_client, avg_KL = get_KL_value(sorted_train_dataset, num_of_CIFAR10_label, num_of_client)

    Save_KL_Result("FL_non_IID_Heuristic_cifar10(CNN)", KL_of_each_client, avg_KL)

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
    model = CIFAR10_CNN()
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    assert (global_model.fl_get_module() == model)

    if cuda_enabled:
        global_model.fl_cuda()
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

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

    Save_Accuracy_of_each_epoch(1, "FL_non_IID_Heuristic_cifar10(CNN)", accuracy_of_each_epoch,
                                best_accuracy_of_each_epoch)
    client_num = num_of_original_client
    global total_execution_time
    CreateResultData("FL_non_IID_Heuristic_cifar10(CNN)", "CIFAR10", "CNN", "non-IID -> IID", client_num,int(trainer_config.epochs), eval_score['Accuracy'], total_execution_time)


@hydra.main(config_path="configs", config_name="cifar10_config", version_base="1.2")
def run(cfg: DictConfig) -> None:
    print('-------------------FL_non_IID_Heuristic_cifar10(CNN)-------------------')
    trainer_config = cfg.trainer
    data_config = cfg.data
    main(
        trainer_config,
        data_config
    )


if __name__ == "__main__":
    f = open('configs/ILP_Heuristic_cifar10_config.json')
    data = json.load(f)
    json_cfg = fl_config_from_json(data)
    cfg = maybe_parse_json_config()
    cfg = OmegaConf.create(json_cfg)
    run(cfg)