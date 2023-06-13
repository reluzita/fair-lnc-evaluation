import pandas as pd
import random
import os
from .data import Dataset

def get_noisy_labels(noise_type, noise_rate, dataset:Dataset, set, store_labels=False):
    """
    Retrieve or generate noisy labels.

    Parameters
    ----------
    noise_type : str
        Type of noise to inject. One of 'random', 'flip', 'bias', 'balanced_bias'.
    noise_rate : float
        Amount of noise to inject.
    dataset : Dataset
        The dataset to use.
    set : str
        The set to use. One of 'train', 'test'.
    store_labels : bool, optional
        Whether to store the noisy labels in a csv file, by default False

    Returns
    -------
    y_noisy: pd.Series
        The noisy labels.
    """
    if store_labels:
        dir = f'data/{dataset.name}_{dataset.sensitive_attr}/{noise_type}/{noise_rate}'

        if os.path.exists(f'{dir}/{set}_labels.csv'):
            return pd.read_csv(f'{dir}/{set}_labels.csv', index_col=0)[dataset.target]

    y_noisy = inject_noise(dataset.get_labels(set), dataset.get_sensitive_attr(set), noise_type, noise_rate)

    if store_labels:
        if not os.path.exists(dir):
            os.makedirs(dir)

        y_noisy.to_csv(f'{dir}/{set}_labels.csv', index=True)

    return y_noisy

def inject_noise(y:pd.Series, group:pd.Series, noise_type, noise_rate):
    """
    Inject noise into clean labels

    Parameters
    ----------
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    noise_type : str
        Type of noise to inject
    noise_rate : float
        Amount of noise to inject

    Returns
    -------
    pd.Series
        Noisy labels
    """
    if noise_type == 'random':
        return random_noise(y, noise_rate)
    elif noise_type == 'flip':
        return flip_noise(y, group, noise_rate)
    elif noise_type == 'bias':
        return bias_noise(y, group, noise_rate)
    elif noise_type == 'balanced_bias':
        return balanced_bias_noise(y, group, noise_rate)
    else:
        raise Exception('Invalid noise type')

def random_noise(y:pd.Series, noise_rate):
    """
    Generate random noise.

    Parameters
    ----------
    y : pd.Series
        Clean labels
    noise_rate : float
        Amount of noise to inject

    Returns
    -------
    y_noisy : pd.Series
        Noisy labels
    """
    random.seed(0)
    y_noisy = y.copy()

    change = random.sample(list(y.index), int(noise_rate * len(y)))
    for i in change:
        y_noisy.loc[i] = 1 - y.loc[i]

    return y_noisy

def flip_noise(y:pd.Series, group:pd.Series, noise_rate):
    """
    Generate group-dependant noise by label flipping. 

    Parameters
    ----------
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    noise_rate : float
        Amount of noise to inject

    Returns
    -------
    y_noisy : pd.Series
        Noisy labels
    """
    random.seed(0)
    y_noisy = y.copy()

    for i in group.loc[group == 1].index:
        if random.random() < noise_rate:
            y_noisy.loc[i] = 1 - y.loc[i] 

    return y_noisy

def bias_noise(y:pd.Series, group:pd.Series, noise_rate):
    """
    Generate group-dependant noise that is biased towards favoring the protected group.

    Parameters
    ----------
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    noise_rate : float
        Amount of noise to inject

    Returns 
    -------
    y_noisy : pd.Series
        Noisy labels
    """
    random.seed(0)
    y_noisy = y.copy()

    for i in group.loc[group == 1].index:
        if random.random() < noise_rate:
            y_noisy.loc[i] = 1

    return y_noisy

def balanced_bias_noise(y:pd.Series, group:pd.Series, noise_rate):
    """
    Generate group-dependant noise that is biased towards simultaneously favoring the protected group and harming the unprotected group.

    Parameters
    ----------
    y : pd.Series
        Clean labels
    group : pd.Series
        Sensitive attribute values
    noise_rate : float
        Amount of noise to inject

    Returns
    -------
    y_noisy : pd.Series
        Noisy labels
    """
    random.seed(0)
    y_noisy = y.copy()

    for i in y_noisy.index:
        if random.random() < noise_rate:
            if group.loc[i] == 1:
                y_noisy.loc[i] = 1
            else:
                y_noisy.loc[i] = 0

    return y_noisy