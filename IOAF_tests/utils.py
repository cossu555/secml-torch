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
from robustbench.utils import download_gdrive
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

def set_MNIST(dataset_path="data", subset_size=None):
    transform = transforms.ToTensor()
    mnist_classes = [str(i) for i in range(10)]

    dataset = torchvision.datasets.MNIST(
        root=dataset_path,
        train=False,
        download=True,
        transform=transform,
    )

    if subset_size is not None:
        dataset = Subset(dataset, list(range(subset_size)))

    return dataset, mnist_classes

def set_dataloader(dataset, num_samples, n_batches):
    batch_size = num_samples // n_batches
    test_subset = Subset(dataset, list(range(num_samples)))
    Data_Loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return Data_Loader


from pathlib import Path
import torch

def set_model(name=None,
    dataset=None,threat_model=None,

    model_class=None,
    model_id=None,
    model_path=None,
    wrap=True,
    device=None,
):
    # Caso 1: modello standard
    if name is not None:
        model_unwrapped = load_model(
            model_name=name,
            dataset=dataset,
            threat_model=threat_model
        )
        model_unwrapped.eval()
        model_unwrapped.to(device)
        return BasePytorchClassifier(model_unwrapped) if wrap else model_unwrapped

    # Caso 2: modello custom
    if model_class is not None and model_path is not None:
        model_unwrapped = model_class().to(device)

        model_path = Path(model_path)

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_id is None:
                raise FileNotFoundError(
                    f"Checkpoint non trovato in {model_path} e model_id non fornito."
                )
            download_gdrive(model_id, model_path)

        state_dict = torch.load(model_path, map_location=device)
        model_unwrapped.load_state_dict(state_dict)
        model_unwrapped.eval()

        return BasePytorchClassifier(model_unwrapped) if wrap else model_unwrapped

    return None

def set_dataset(dataset_name, dataset_path="data", subset_size=None):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return set_CIFAR_10()
    elif dataset_name == "mnist":
        return set_MNIST(dataset_path=dataset_path, subset_size=subset_size)
    else:
        raise ValueError(f"Dataset '{dataset_name}' non supportato")

def setup(
    num_samples,
    n_batches,
    num_steps,
    step_size,
    epsilon,
    model=None,
    dataset_name="cifar10",
    threat_model=None,
    perturbation_model=LpPerturbationModels.LINF,
    y_target=None,
    dataset_path="data",
    subset_size=None,
    model_class=None,
    model_id=None,
    model_path=None,
):
    trackers = set_trackers()

    # dataset corretto in base al nome
    dataset, _ = set_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        subset_size=subset_size,
    )
    dataloader = set_dataloader(dataset, num_samples, n_batches)

    # caricamento modello:
    # - se model è già un oggetto modello, lo usa
    # - se model è una stringa, la interpreta come nome del modello
    if isinstance(model, str) or model is None:
        model = set_model(
            name=model,
            dataset=dataset_name,
            threat_model=threat_model,
            model_class=model_class,
            model_id=model_id,
            model_path=model_path,
        )
    else:
        # se passi un modello torch già istanziato, lo wrappa solo se serve
        if hasattr(model, "eval"):
            model.eval()
        if not isinstance(model, BasePytorchClassifier):
            model = BasePytorchClassifier(model)

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

