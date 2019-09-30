import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import torch


def get_device(opt, input_tensor):
    if opt.device.type != "cpu":
        cur_device = input_tensor.get_device()
    else:
        cur_device = opt.device

    return cur_device


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def scatter(opt, x, colors, label):
    """
    creates scatter plot for t-SNE visualization
    :param x: 2-D latent space as output by t-SNE
    :param colors: labels for each datapoint in x, used to assign different colors to them
    :param idx: used for naming the file, to be able to track progress throughout training
    """
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.ravel().astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")

    plt.savefig(
        os.path.join(opt.log_path_latent, "latent_space_{}.png".format(label)), dpi=120
    )
    plt.close()


def fit_TSNE_and_plot(opt, feature_space, speaker_labels, label):
    feature_space = np.reshape(
        feature_space, (np.shape(feature_space)[0] * np.shape(feature_space)[1], -1)
    )
    speaker_labels = np.reshape(speaker_labels, (-1, 1))

    # X: array, shape(n_samples, n_features)
    projection = TSNE().fit_transform(feature_space)

    scatter(opt, projection, speaker_labels, label)
