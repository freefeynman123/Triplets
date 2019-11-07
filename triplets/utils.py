import torch


def freeze_layers(
        model: torch.nn.Sequential,
        n_layers_to_train: int
) -> torch.nn.Sequential:
    """
    Function to freeze given number of layers for selected model
    :param model: Instance of Pytorch model
    :param n_layers_to_train: number of layers to train, counting from the last one.
                              The rest of the layers is going to be frozen.
    :return: Model with frozen layers.
    """
    n_layers = len(list(model.children()))
    for idx, child in enumerate(model.children()):
        if idx < (n_layers - n_layers_to_train):
            for param in child.parameters():
                param.requires_grad = False
    return model
