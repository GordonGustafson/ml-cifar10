import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


import collections
import copy
from dataclasses import dataclass
import functools
import math
import PIL
import random
import time
from typing import Callable, List, Dict

################################################################################

device = torch.device('cuda')
dtype = torch.float32

USE_HFLIP_IN_ACCURACY_EVAL = False

################################################################################

NUM_BUCKETS = 200
MIN = -10
MAX = 10
BAR_X_POSITIONS = np.linspace(start=MIN, stop=MAX, num=NUM_BUCKETS+1)[:-1]

def plot_histogram(axes, frequencies):
    axes.bar(x=BAR_X_POSITIONS, height=frequencies, align='edge', width=(MAX-MIN)/NUM_BUCKETS)

def record_outputs_in_eval(output_frequencies, module, input, output):
    if not module.training:
        # This does an inplace update
        output_frequencies += torch.histc(output, bins=NUM_BUCKETS, min=MIN, max=MAX)

################################################################################

class ToFloat16(object):
    def __repr__(self):
        return f"ToFloat16()"

    def __call__(self, tensor):
        return tensor.type(torch.float16)

# Subtract hardcoded RGB means and divide by hardcoded RGB stds.
MEAN_PIXEL_0_1 = (0.4914, 0.4822, 0.4465)
MEAN_PIXEL_0_255 = tuple([round(255 * v) for v in MEAN_PIXEL_0_1])
hardcoded_normalize = T.Normalize(MEAN_PIXEL_0_1, (0.2023, 0.1994, 0.2010))

std_transform = T.Compose([
    T.ToTensor(),
    hardcoded_normalize,
])

hflip_std = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    hardcoded_normalize,
])

vflip_std = T.Compose([
    T.RandomVerticalFlip(p=0.5),
    T.ToTensor(),
    hardcoded_normalize,
])

hflip_vflip_std = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ToTensor(),
    hardcoded_normalize,
])

def get_random_crop_transform(p_crop, crop_size):
    return T.Compose([
        T.RandomApply([T.RandomCrop(crop_size), T.Resize((32, 32))], p=p_crop),
        T.ToTensor(),
        hardcoded_normalize,
    ])

def get_transform_hflip_crop(p_crop, crop_size):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomCrop(crop_size), T.Resize((32, 32))], p=p_crop),
        T.ToTensor(),
        hardcoded_normalize,
    ])

class RandomWeightedChoice(object):
    def __init__(self, population, weights):
        self.population = population
        self.weights = weights

    def __call__(self, img):
        transform = random.choices(population=self.population,
                                   weights=self.weights, k=1)
        return transform[0](img)

    def __repr__(self):
        return f"RandomWeightedChoice({self.population}, {self.weights})"

def get_transform_hflip_weighted_crop(crop_weights, crop_sizes):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        RandomWeightedChoice(population=[T.RandomCrop(size) for size in crop_sizes],
                             weights=crop_weights),
        T.Resize((32, 32)),
        T.ToTensor(),
        hardcoded_normalize,
    ])

def get_transform_hflip_weighted_perspective_crop(crop_weights, crop_sizes,
                                                  p_perspective, distortion_scale, interpolation, fill):
    # RandomPerspective transforms only include `p` in their __repr__ method, so
    # you'll want to include `distortion_scale` in TrainParams if you use this,
    # because otherwise it won't be recorded anywhere.
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPerspective(distortion_scale=distortion_scale,
                            p=p_perspective,
                            interpolation=interpolation,
                            fill=fill),
        RandomWeightedChoice(population=[T.RandomCrop(size) for size in crop_sizes],
                             weights=crop_weights),
        T.Resize((32, 32)),
        T.ToTensor(),
        hardcoded_normalize,
    ])

def get_transform_hflip_rotate_weighted_crop(crop_weights, crop_sizes,
                                             p_rotation, degrees, rotation_resample=PIL.Image.NEAREST):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=degrees, resample=rotation_resample, fill=MEAN_PIXEL_0_255)], p=p_rotation),
        RandomWeightedChoice(population=[T.RandomCrop(size) for size in crop_sizes],
                             weights=crop_weights),
        T.Resize((32, 32)),
        T.ToTensor(),
        hardcoded_normalize,
    ])


def get_transform_hflip_weighted_crop_colorjitter(crop_weights, crop_sizes, brightness, contrast, saturation, hue):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        RandomWeightedChoice(population=[T.RandomCrop(size) for size in crop_sizes], weights=crop_weights),
        T.Resize((32, 32)),
        T.ColorJitter(brightness, contrast, saturation, hue),
        T.ToTensor(),
        hardcoded_normalize,
    ])

class JPGEncodeDecode(object):
    def __init__(self, jpg_quality):
        self.jpg_quality = jpg_quality

    def __call__(self, img):
        # From https://stackoverflow.com/questions/40768621/python-opencv-jpeg-compression-in-memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
        image_as_np = np.array(img, dtype=np.uint8)
        _, encoded_image = cv2.imencode('.jpg', image_as_np, encode_param)
        return cv2.imdecode(encoded_image, 1)

    def __repr__(self):
        return f"JPGEncodeDecode({self.jpg_quality})"

def get_transform_hflip_crop_jpg(p_crop, crop_size, p_jpg, jpg_quality):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomCrop(crop_size), T.Resize((32, 32))], p=p_crop),
        T.RandomApply([JPGEncodeDecode(jpg_quality)], p=p_jpg),
        T.ToTensor(),
        hardcoded_normalize,
    ])

def get_24_28_32_random_crop_transform_with_hflip():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomChoice([
            T.Compose([T.RandomCrop((28, 28)), T.Resize((32, 32))]),
            T.Compose([T.RandomCrop((24, 24)), T.Resize((32, 32))]),
            lambda x: x
        ]),
        T.ToTensor(),
        hardcoded_normalize,
    ])

def gaussian_noise_adder(std, size):
    def add_gaussian_noise(data):
        noise = std * torch.randn(size)
        return data + noise

    return add_gaussian_noise

class GaussianNoiseTransform(object):
    def __init__(self, std, size):
        super(GaussianNoiseTransform, self).__init__()
        self.std = std
        self.size = size

    def __call__(self, tensor):
        noise = self.std * torch.randn(self.size)
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std}, size={self.size})\n"

def get_random_crop_transform_with_hflip_with_noise(p_crop, crop_size, p_noise, noise_std):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomCrop(crop_size), T.Resize((32, 32))], p=p_crop),
        T.ToTensor(),
        hardcoded_normalize,
        T.RandomApply([GaussianNoiseTransform(noise_std, (32, 32))], p=p_noise),
    ])

BATCH_SIZE = 64

def get_datasets(transform):
    augmented = dset.CIFAR10('./datasets/cifar-10', train=True, download=True, transform=transform)
    if USE_HFLIP_IN_ACCURACY_EVAL:
        non_augmented = dset.CIFAR10('./datasets/cifar-10', train=True, download=True, transform=hflip_std)
    else:
        non_augmented = dset.CIFAR10('./datasets/cifar-10', train=True, download=True, transform=std_transform)

    return {"augmented": augmented,
            "non_augmented": non_augmented}

def get_dataloaders(datasets, num_train, num_val, random_seed):
    augmented = datasets["augmented"]
    non_augmented = datasets["non_augmented"]
    # all_keys_and_ys = [(index, y) for (image, y), index in enumerate(non_augmented)]
    all_keys = range(0, num_train + num_val)
    all_ys = [non_augmented[key][1] for key in all_keys]
    train_keys, val_all_keys = sklearn.model_selection.train_test_split(all_keys, test_size=num_val, stratify=all_ys, random_state=0)

    val_all_ys = [non_augmented[key][1] for key in val_all_keys]
    val_1_keys, val_2_keys = sklearn.model_selection.train_test_split(val_all_keys, test_size=num_val // 2, stratify=val_all_ys, random_state=0)

    return {
        "train_augmented": DataLoader(augmented, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(train_keys)),
        "train_non_augmented": DataLoader(non_augmented, batch_size=BATCH_SIZE, sampler=train_keys),
        "val_1": DataLoader(non_augmented, batch_size=BATCH_SIZE, sampler=val_1_keys),
        "val_2": DataLoader(non_augmented, batch_size=BATCH_SIZE, sampler=val_2_keys),
        "val_all": DataLoader(non_augmented, batch_size=BATCH_SIZE, sampler=val_all_keys),
    }

################################################################################

class NegativeRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -F.relu(-x)

def bent_relu(x, threshold_x):
    torch.relu_(x)  # in place relu, likely faster
    return torch.where(x < threshold_x, x / 2, x - (threshold_x / 2))

class BentReLU(nn.Module):
    def __init__(self, threshold_x):
        super().__init__()
        self.threshold_x = threshold_x

    def forward(self, x):
        return bent_relu(x, self.threshold_x)

# Copied from https://discuss.pytorch.org/t/kronecker-product/3919/10
def kronecker_product(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

class ClassSpecificFilters(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        assert input_size % output_size == 0
        filters_per_class = input_size // output_size
        self.weights = kronecker_product(torch.eye(output_size),
                                         torch.ones(filters_per_class, 1))
        self.weights = self.weights.to(device=device)

    def forward(self, x):
        return torch.matmul(x, self.weights)

class ElementwiseProduct(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_size, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.weights

class FilterBias(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((1, num_filters, 1, 1), dtype=dtype, device=device))

    def forward(self, x):
        return x + self.weights

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def froz_conv(*args, **kwargs):
    weights = kwargs.pop("weights")
    layer = nn.Conv2d(*args, **kwargs)
    with torch.no_grad():
        layer.weight = nn.Parameter(weights, requires_grad=False)
        for param in layer.parameters():
            param.requires_grad = False
    return layer

class RunAndConcat2D(nn.Module):
    def __init__(self, modules_list):
        super().__init__()
        self.modules_list = modules_list
        for idx, module in enumerate(self.modules_list):
            self.add_module(str(idx), module)

    def forward(self, x):
        outputs = [m.forward(x) for m in self.modules_list]
        return torch.cat(outputs, dim=1)

class ContrastiveCenterPool2d(nn.Module):
    """
    Contrastive center pooling who's filter averages to 1, like average pooling does.
    """
    def __init__(self):
        super().__init__()
        self.center_weights = torch.tensor([[-0.25, -0.25, -0.25, -0.25],
                                            [-0.25,     1,     1, -0.25],
                                            [-0.25,     1,     1, -0.25],
                                            [-0.25, -0.25, -0.25, -0.25]],
                                           device=device, dtype=dtype)

    def forward(self, x):
        center_product = torch.einsum("nchw,hw->nc", x, self.center_weights)
        return center_product.unsqueeze(2).unsqueeze(3)

class OrthoCenterPool2d(nn.Module):
    """
    Center pooling who's filter is orthogonal to Average pooling
    """
    def __init__(self):
        super().__init__()
        self.center_weights = torch.tensor([[-0.3333, -0.3333, -0.3333, -0.3333],
                                            [-0.3333,       1,       1, -0.3333],
                                            [-0.3333,       1,       1, -0.3333],
                                            [-0.3333, -0.3333, -0.3333, -0.3333]],
                                           device=device, dtype=dtype)

    def forward(self, x):
        center_product = torch.einsum("nchw,hw->nc", x, self.center_weights)
        return center_product.unsqueeze(2).unsqueeze(3)

class CenterPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_weights = torch.tensor([[0, 0, 0, 0],
                                            [0, 1, 1, 0],
                                            [0, 1, 1, 0],
                                            [0, 0, 0, 0]],
                                           device=device, dtype=dtype) * (1/4)

    def forward(self, x):
        center_product = torch.einsum("nchw,hw->nc", x, self.center_weights)
        return center_product.unsqueeze(2).unsqueeze(3)

class EdgePool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_weights = torch.tensor([[1, 1, 1, 1],
                                          [1, 0, 0, 1],
                                          [1, 0, 0, 1],
                                          [1, 1, 1, 1]],
                                         device=device, dtype=dtype) * (1/12)

    def forward(self, x):
        edge_product = torch.einsum("nchw,hw->nc", x, self.edge_weights)
        return edge_product.unsqueeze(2).unsqueeze(3)

# torch's schedules derive from _LRScheduler
# This looks to solve the same problem, but it wasn't particularly clear:
# https://pytorch.org/ignite/contrib/handlers.html#ignite.contrib.handlers.param_scheduler.ConcatScheduler
class ConcatLRSchedulers(object):
    def __init__(self, optimizer, schedulers, epochs_per_scheduler, initial_lrs):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.epochs_per_scheduler = epochs_per_scheduler
        # NOTE: first element is ignored (set it on the optimizer)
        self.initial_lrs = initial_lrs

        self.epochs_on_current_scheduler = 0
        self.current_scheduler_index = 0

    def current_scheduler(self):
        return self.schedulers[self.current_scheduler_index]

    def step_batch(self):
        if isinstance(self.current_scheduler(), optim.lr_scheduler.OneCycleLR):
            self.current_scheduler().step()

    def step_epoch(self):
        self.epochs_on_current_scheduler += 1
        if self.epochs_on_current_scheduler == self.epochs_per_scheduler[self.current_scheduler_index]:
            self.epochs_on_current_scheduler = 0
            self.current_scheduler_index += 1

            if self.current_scheduler_index < len(self.initial_lrs):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.initial_lrs[self.current_scheduler_index]
        else:
            if not isinstance(self.current_scheduler(), optim.lr_scheduler.OneCycleLR):
                self.current_scheduler().step()

################################################################################

def learned_features(model, loader, dataset_size):
    # hardcoding lots of stuff for now
    NUM_FEATURES = 256
    learned_features = np.zeros((dataset_size, NUM_FEATURES), dtype=np.float32)
    ground_truth = np.zeros((dataset_size,), dtype=np.float32)

    # Assumes that there's only one classification layer
    model_without_classification_layer = nn.Sequential(*list(model.children())[:-1])
    model.eval()  # not sure if necessary
    model_without_classification_layer.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            learned_features_for_batch = model_without_classification_layer(x).cpu().numpy()
            min_index = batch_index * BATCH_SIZE
            max_index = min(min_index + BATCH_SIZE, dataset_size)
            learned_features[min_index:max_index, :] = learned_features_for_batch
            ground_truth[min_index:max_index] = y.cpu().numpy()

    return learned_features, ground_truth

################################################################################

def compute_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    return float(num_correct) / num_samples

def init_weights(model):
    if type(model) in [nn.Conv2d, nn.Linear] and any(p.requires_grad for p in model.parameters()):
        nn.init.kaiming_normal_(model.weight)

# LRParams = collections.namedtuple("LRParams", ["init_lr", "step_size", "gamma"])

@dataclass
class ComponentParams(object):
    weight_decay: float
    optimizer: str
    optimizer_params: Dict
    lr_scheduler: str
    lr_scheduler_params: Dict

    def _key(self):
        return (self.weight_decay,
                self.optimizer,
                frozenset(self.optimizer_params.items()),
                self.lr_scheduler,
                frozenset(self.lr_scheduler_params.items()))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        # TODO: is `self.__class__` okay here?
        if isinstance(other, self.__class__):
            return self._key() == other._key()
        return NotImplemented

@dataclass
class TrainParams(object):
    transform: Callable
    use_hflip_in_accuracy_eval: bool
    num_train: int
    num_val: int
    random_seed: int
    num_epochs: int
    # This is only here because RandomPerspective transforms only include `p` in
    # their __repr__ method, so it won't be included otherwise.
    distortion_scale: float
    # TODO: find out how to better fit p_dropout in.
    # Right now it's never used except in _key to make equality work.
    p_dropout: float

    head_params: ComponentParams
    backbone_params: ComponentParams

    def _key(self):
        # Pytorch transforms implement repr
        return (repr(self.transform),
                self.use_hflip_in_accuracy_eval,
                self.num_train,
                self.num_val,
                self.random_seed,
                self.num_epochs,
                self.distortion_scale,
                self.p_dropout,
                self.head_params._key(),
                self.backbone_params._key())

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        # TODO: is `self.__class__` okay here?
        if isinstance(other, self.__class__):
            return self._key() == other._key()
        return NotImplemented

def get_optimizer(parameters, component_params):
    optimization_params = [{"params": parameters, "weight_decay": component_params.weight_decay}]
    if component_params.optimizer == "SGD":
        optimizer = optim.SGD(params=optimization_params, **component_params.optimizer_params)
    elif component_params.optimizer == "Adam":
        optimizer = optim.Adam(params=optimization_params, **component_params.optimizer_params)
    elif component_params.optimizer == "AdamW":
        optimizer = optim.AdamW(params=optimization_params, **component_params.optimizer_params)
    else:
        raise ValueError(f"unknown optimizer: {train_params.optimizer}")
    return optimizer

def get_scheduler(optimizer, component_params):
    cp = component_params
    if cp.lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **cp.lr_scheduler_params)
    elif cp.lr_scheduler == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **cp.lr_scheduler_params)
    # elif cp.lr_scheduler == "ConcatLRSchedulers":
    #     schedulers = [get_scheduler(optimizer, name, params) for name, params in cp.scheduler_params["schedulers"]]
    #     scheduler = ConcatLRSchedulers(optimizer,
    #                                    schedulers=schedulers,
    #                                    epochs_per_scheduler=cp.scheduler_params["epochs_per_scheduler"],
    #                                    initial_lrs=cp.scheduler_params["initial_lrs"])
    else:
        raise ValueError(f"unknown lr_scheduler: {cp.lr_scheduler}")

    return scheduler

def get_transform_by_schedule(train_params, epoch):
    portion_through_training = epoch / train_params.num_epochs

    # Assume that both head and backbone optimizers have some "pct_start".
    peak_of_sine_curve = train_params.head_params.lr_scheduler_params["pct_start"]
    if portion_through_training < peak_of_sine_curve:
        portion_through_sine = 0.5 * portion_through_training / peak_of_sine_curve
    else:
        portion_through_sine = 0.5 + 0.5 * (portion_through_training - peak_of_sine_curve) / (1 - peak_of_sine_curve)

    # Only go through half of sine's cycle
    sine_value = math.sin(math.pi * portion_through_sine)
    # NOTE: need to restore `cropped_portion_multiplier` field in TrainParams to use this function again
    cropped_percentage = min(1, train_params.cropped_portion_multiplier * sine_value)
    uncropped_percentage = 1 - cropped_percentage
    print(f"uncropped_percentage: {uncropped_percentage}")
    # crop_aug_args = {"p_rotation": cropped_percentage,
    #                  "degrees": 10,
    #                  "crop_weights": [cropped_percentage / 2,
    #                                   0.0,
    #                                   cropped_percentage / 2,
    #                                   uncropped_percentage],
    #                  "crop_sizes": [(28, 28), (32, 24), (24, 32), (32, 32)]}
    # transform = get_transform_hflip_rotate_weighted_crop(**crop_aug_args)

    crop_aug_args = {"crop_weights": [cropped_percentage / 2,
                                      0.0,
                                      cropped_percentage / 2,
                                      uncropped_percentage],
                     "crop_sizes": [(28, 28), (32, 24), (24, 32), (32, 32)],
                     "brightness": sine_value/2,
                     "contrast": sine_value/2,
                     "saturation": sine_value/2,
                     "hue": sine_value/2,
    }
    transform = get_transform_hflip_weighted_crop_colorjitter(**crop_aug_args)

    return transform

@dataclass
class TrainResults:
    model_name: str
    train_params: TrainParams
    model_history: List[nn.Module]
    train_accuracy_history: List[float]
    val1_accuracy_history: List[float]
    val2_accuracy_history: List[float]
    val_accuracy_history: List[float]
    layer_output_frequencies: torch.Tensor
    index_of_best_val: int

def index_of_max(lst):
    return lst.index(max(lst))

def train_model(model_name, model, train_params):
    datasets = get_datasets(train_params.transform)
    loaders = get_dataloaders(datasets,
                              num_train=train_params.num_train,
                              num_val=train_params.num_val,
                              random_seed=train_params.random_seed)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model.apply(init_weights)

    head_optimizer = get_optimizer(model.head.parameters(), train_params.head_params)
    head_scheduler = get_scheduler(head_optimizer, train_params.head_params)
    backbone_optimizer = get_optimizer(model.backbone.parameters(), train_params.backbone_params)
    backbone_scheduler = get_scheduler(backbone_optimizer, train_params.backbone_params)

    model_history = []
    train_accuracy_history = []
    val1_accuracy_history = []
    val2_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(train_params.num_epochs):
        loop_start_time = time.time()

        model.train()  # put model to training mode
        for x, y in loaders["train_augmented"]:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            head_optimizer.zero_grad()
            backbone_optimizer.zero_grad()

            loss.backward()

            head_optimizer.step()
            backbone_optimizer.step()

            # TODO: scheduler.step here only for OneCycleLR
            head_scheduler.step()
            backbone_scheduler.step()
        loop_end_time = time.time()
        epoch_time = loop_end_time - loop_start_time

        is_last_epoch = epoch == train_params.num_epochs - 1
        if is_last_epoch:
            # Add instrumentation for layer activations
            layers = model.get_layers()
            layer_output_frequencies = [torch.zeros(NUM_BUCKETS, device=device) for l in layers]
            for layer, output_frequencies in zip(layers, layer_output_frequencies):
                layer.register_forward_hook(functools.partial(record_outputs_in_eval, output_frequencies))

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            train_accuracy = compute_accuracy(loaders["train_non_augmented"], model)
            val1_accuracy = compute_accuracy(loaders["val_1"], model)
            val2_accuracy = compute_accuracy(loaders["val_2"], model)
            val_accuracy = round((val1_accuracy + val2_accuracy) / 2, 4)

        current_head_lr = head_optimizer.param_groups[0]['lr']
        current_head_momentum = head_optimizer.param_groups[0]['betas'][0]
        print(f"After epoch {epoch} with head lr={current_head_lr}, momentum={current_head_momentum}: train accuracy: {round(train_accuracy, 3)}  val accuracy: {val_accuracy}  epoch duration: {epoch_time}")
        train_accuracy_history.append(train_accuracy)
        val1_accuracy_history.append(val1_accuracy)
        val2_accuracy_history.append(val2_accuracy)
        val_accuracy_history.append(val_accuracy)
        model_history.append(copy.deepcopy(model))

        # TODO: scheduler.step here for non-OneCycleLR
        # scheduler.step()

    index_of_best_val = index_of_max(val_accuracy_history)
    return TrainResults(model_name=model_name,
                        train_params=train_params,
                        model_history=model_history,
                        train_accuracy_history=train_accuracy_history,
                        val1_accuracy_history=val1_accuracy_history,
                        val2_accuracy_history=val2_accuracy_history,
                        val_accuracy_history=val_accuracy_history,
                        layer_output_frequencies=layer_output_frequencies,
                        index_of_best_val=index_of_best_val)


def fill_results_cache(models_dict, model_results_cache):
    for model_def, model in models_dict.items():
        if model_def not in model_results_cache:
            model_name, train_params = model_def
            print(f"Training {model_def}")
            result = train_model(model_name, model, train_params)
            model_results_cache[model_def] = result

class TwoStageModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        backbone_result = self.backbone.forward(x)
        return self.head.forward(backbone_result)

    def get_layers(self):
        # Assumes backbone is an nn.Sequential
        backbone_layers = list(self.backbone.modules())[1:]
        # Assumes head is a single layer
        head_layers = [self.head]
        return backbone_layers + head_layers



