import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_reduction_results(reduced_features: np.ndarray, hue, label_names: str):
    """
    Function to plot results from tSNE dimensionality reduction.
    It accounts for different number of neighbors
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(17, 10))
    legend = 'full'
    g = sns.scatterplot(x=reduced_features.T[0], y=reduced_features.T[1], hue=hue, legend=legend,
                        palette=sns.color_palette("Set1", n_colors=10), ax=ax)
    legend_handles = g.get_legend_handles_labels()[0]
    g.legend(legend_handles, label_names)
