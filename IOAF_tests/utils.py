# Import tracking components
from secmlt.trackers import (
    LossTracker,
    PredictionTracker,
    PerturbationNormTracker,
    GradientNormTracker,
    TensorboardTracker,
)
#dataset
import torchvision
import torchvision.transforms as transforms
#dataloader
from torch.utils.data import DataLoader, Subset
#model
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from robustbench import load_model
#attack
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD
from secmlt.adv.backends import Backends

dataset_path = "data/datasets/"
logs_path = "data/logs/pgd_tutorial"


def set_trackers():
    trackers = [
        LossTracker(),
        PredictionTracker(),
        PerturbationNormTracker("linf"),
        GradientNormTracker(),
    ]
    return trackers


def set_CIFAR_10():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_classes = [
        "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
    ]

    Dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=False,
        download=True,
        transform=transform
    )
    return Dataset, cifar10_classes


def set_dataloader(dataset, num_samples, n_batches):
    batch_size = num_samples // n_batches
    test_subset = Subset(dataset, list(range(num_samples)))
    Data_Loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return Data_Loader


def set_model(name, dataset, threat_model):
    if name is None:
        return None
    model_unwrapped = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    model_unwrapped.eval()
    # wrapping of de model
    Model = BasePytorchClassifier(model_unwrapped)
    return Model

def setup(num_samples, n_batches, num_steps, step_size, epsilon,model ,dataset_name,threat_model, perturbation_model=LpPerturbationModels.LINF,
              y_target=None):
    trackers = set_trackers()
    dataset, _ = set_CIFAR_10()
    dataloader = set_dataloader(dataset, num_samples, n_batches)
    model = set_model(model,dataset_name , threat_model)

    tensorboard_tracker = TensorboardTracker(logs_path, trackers)
    attack = PGD(
        perturbation_model=perturbation_model,
        epsilon=epsilon,
        num_steps=num_steps,
        step_size=step_size,
        random_start=False,
        y_target=y_target,
        backend=Backends.NATIVE,
        trackers=tensorboard_tracker,
    )
    return attack, model, dataloader

