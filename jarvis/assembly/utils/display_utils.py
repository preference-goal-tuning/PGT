from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from scipy import stats
from sklearn.manifold import TSNE
from typing import Iterable, Literal, Optional, Union
import torch
import random
import seaborn as sns
from sklearn.linear_model import LinearRegression


markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']


def tsne_all(data: Iterable[Iterable[Union[np.array, torch.tensor]]], save_path, labels:Optional[Iterable[str]]=None, title=None, perplexity=30, learning_rate='auto', n_iter=300):
    num_sets = len(data)
    if labels is not None:
        assert len(labels) == num_sets
    
    data = [deepcopy(group) for group in data]
    np.random.seed(0)
    random.seed(0)
    
    datadim = 1
    if len(data[0][0].shape) != 0:
        for d in data[0][0].shape:
            datadim *= d
    for i in range(num_sets):
        if isinstance(data[i], np.ndarray):
            data[i] = data[i].reshape((len(data[i]), datadim))
        elif isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()
            data[i] = data[i].reshape((len(data[i]), datadim))
        else:
            for j in range(len(data[i])):
                if isinstance(data[i][j], torch.Tensor):
                    data[i][j] = data[i][j].cpu().numpy().reshape((datadim,))
            data[i] = np.vstack(data[i])
    classes = [np.ones(shape=(data[i].shape[0])) * i for i in range(num_sets)]
    
    data = np.vstack(data)
    classes = np.hstack(classes)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
    low_dim_data = tsne.fit_transform(data)
    plt.figure(figsize=(8, 6))
    for i in range(num_sets):
        if labels is not None:
            plt.scatter(low_dim_data[classes == i, 0], low_dim_data[classes == i, 1], c=colors[i], marker=markers[i], label=labels[i])
        else:
            plt.scatter(low_dim_data[classes == i, 0], low_dim_data[classes == i, 1], c=colors[i], marker=markers[i])
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.savefig(save_path)


def plot_linear_reg(x, ys, save_path, labels=None, xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=(10, 6))
    if labels is None:
        for i, y in enumerate(ys):
            sns.regplot(x=x, y=y, color=colors[i], marker=markers[i], line_kws={'linestyle': '--', 'color': colors[i], 'linewidth': 1})
    else:
        assert len(ys) == len(labels)
        for i, (y, label) in enumerate(zip(ys, labels)):
            sns.regplot(x=x, y=y, color=colors[i], marker=markers[i], line_kws={'linestyle': '--', 'color': colors[i], 'linewidth': 1}, label=label)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if labels is not None:
        plt.legend()
    plt.savefig(save_path if save_path.endswith('.png') else save_path + '.png')
    plt.close()


def structify(obj):
    if isinstance(obj, np.ndarray):
        return 'np.ndarray:'+str(tuple(obj.shape))
    elif isinstance(obj, torch.Tensor):
        return 'torch.Tensor:'+str(tuple(obj.shape))
    elif isinstance(obj, (tuple, list)):
        return str(type(obj)) + str([structify(o) for o in obj])
    elif isinstance(obj, torch.nn.Module):
        return {k:type(obj[k]) for k in obj}
    elif not isinstance(obj, (dict, DictConfig)):
        return obj
    else:
        return {k:structify(obj[k]) for k in obj}
    
def print_struct(obj):
    print(structify(obj))


def pdf(data, save_path:str, bw_adjust:float, title=None, xlabel=None, ylabel=None):
    sns.kdeplot(data, fill=True, bw_adjust=bw_adjust)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def multi_plot(ys: Iterable[np.ndarray], save_path, labels=None, xlabel=None, ylabel=None, title=None):
    plt.plot()
    plt.savefig(save_path)
    plt.close()