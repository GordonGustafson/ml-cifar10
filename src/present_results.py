from src.model_utils import TrainResults

import itertools
import matplotlib.pyplot as plt
from typing import List

def total_params(model):
    # from https://stackoverflow.com/a/49201237
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_params(model):
    return [(name, total_params(module)) for name, module in model.named_modules()]

def index_of_max(lst):
    return lst.index(max(lst))

def get_training_results_string(train_results: TrainResults):
    tr = train_results
    train_params_string = str(tr.train_params).replace("\n", "").replace(" ", "")

    best_val_for_model = max(tr.val_accuracy_history)
    epoch_with_best_val = tr.val_accuracy_history.index(best_val_for_model)
    train_for_best_val = tr.train_accuracy_history[epoch_with_best_val]
    duality_gap_for_best_val = train_for_best_val - best_val_for_model

    val2_from_best_val1 = tr.val2_accuracy_history[index_of_max(tr.val1_accuracy_history)]
    val1_from_best_val2 = tr.val1_accuracy_history[index_of_max(tr.val2_accuracy_history)]
    crossvalidated_val = (val2_from_best_val1 + val1_from_best_val2) / 2

    summary = (f"| {tr.model_name}, {train_params_string} | {best_val_for_model} | {round(crossvalidated_val, 4)} | {epoch_with_best_val} |{round(train_for_best_val, 3)} "
               f"| {round(duality_gap_for_best_val, 3)} | {total_params(tr.model_history[0])} | {tr.train_accuracy_history} | {tr.val_accuracy_history}")
    return summary

def get_train_val_accuracies_fig(results_list: List[TrainResults]):
    accuracy_fig, accuracy_axes_grid = plt.subplots(ncols=3, nrows=3, constrained_layout=True, figsize=(20.0, 8.0))
    accuracy_axes_list = list(itertools.chain.from_iterable(accuracy_axes_grid))

    for (index, results) in enumerate(results_list):
        accuracy_axes = accuracy_axes_list[index]
        accuracy_axes.grid()

        accuracy_axes.plot(results.train_accuracy_history, label="training_accuracy")
        accuracy_axes.plot(results.val_accuracy_history, label="validation accuracy")

    return accuracy_fig

# Currently used for visualizing augmented data rather than presenting results
def to_image_grid(images, ncols=3):
    num_images, height, width, channels = images.shape
    nrows = num_images//ncols
    assert num_images == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, channels)
    result = (images.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, channels))
    return result
