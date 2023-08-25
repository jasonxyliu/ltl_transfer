#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:29:47 2022
"""
import os
import dill
import numpy as np

from formula_sampler import sample_formula


ENV2PROPS = {
    "minecraft": ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 's'),
    "spot": ('a', 'b', 'c', 'd', 'j', 'p')
}
SET_TYPES = ('hard', 'soft', 'soft_strict', 'no_orders', 'mixed')
TRAIN_SIZES = (10, 20, 30, 40, 50)
TEST_SIZE = 100


def create_datasets(save_dpath, env_name, props, set_types=SET_TYPES, duplicate_ok=True, train_sizes=TRAIN_SIZES, test_size=TEST_SIZE):
    dataset_dpath = os.path.join(save_dpath, "experiments", "datasets", env_name)
    train_set_dpath = os.path.join(dataset_dpath, 'training')
    os.makedirs(train_set_dpath, exist_ok=True)
    test_set_dpath = os.path.join(dataset_dpath, 'test')
    os.makedirs(test_set_dpath, exist_ok=True)

    for typ in set_types:
        create_dataset(props, train_set_dpath, 'train', typ, duplicate_ok, train_sizes)
        create_dataset(props, test_set_dpath, 'test', typ, duplicate_ok, [test_size])


def create_dataset(props, save_dpath, set_name='train', set_type='mixed', duplicate_ok=True, sizes=TRAIN_SIZES):
    n_formulas = np.max(sizes)
    if duplicate_ok:
        formulas = sample_dataset_formulas(props=props, set_type=set_type, n=n_formulas)
    else:
        formulas = sample_dataset_unique_formulas(props=props, set_type=set_type, n=n_formulas)

    for size in sizes:
        filename = f'{set_name}_{set_type}_{size}.pkl'
        with open(os.path.join(save_dpath, filename), 'wb') as file:
            dill.dump(formulas[0: size], file)
        human_readable_filename = f'{set_name}_{set_type}_{size}.txt'
        with open(os.path.join(save_dpath, human_readable_filename), 'w') as file:
            for idx, formula in enumerate(formulas[0: size]):
                file.write(f"{idx}: {str(formula)}\n")


def sample_dataset_formulas(props, set_type='mixed', n=50):
    """
    Allowed to sample duplicated LTL formulas into the dataset.
    """
    if set_type == 'no_orders':
        formulas = [sample_formula(props=props, orders=False)[0] for _ in range(n)]
    else:
        formulas = [sample_formula(props=props, orders=True, order_type=set_type)[0] for _ in range(n)]
    return formulas


def sample_dataset_unique_formulas(props, set_type='mixed', n=50):
    """
    Only sample unique LTL formulas into the dataset.
    """
    formulas = []
    num_samples_sofar = 0
    while num_samples_sofar < n:
        if set_type == 'no_orders':
            formula = sample_formula(props=props, orders=False)[0]
        else:
            formula = sample_formula(props=props, orders=True, order_type=set_type)[0]
        if formula not in formulas:
            formulas.append(formula)
            num_samples_sofar += 1
    return formulas


def read_train_test_formulas(dataset_dpath, env_name, train_set_type='mixed', test_set_type='hard', train_size=50, test_size=100):
    dataset_dpath = os.path.join(dataset_dpath, "experiments", "datasets", env_name)
    train_set_dpath = os.path.join(dataset_dpath, 'training')
    test_set_dpath = os.path.join(dataset_dpath, 'test')

    train_set_name = f'train_{train_set_type}_{train_size}.pkl'
    test_set_name = f'test_{test_set_type}_{test_size}.pkl'

    with open(os.path.join(train_set_dpath, train_set_name), 'rb') as file:
        train_formulas = dill.load(file)
    with open(os.path.join(test_set_dpath, test_set_name), 'rb') as file:
        test_formulas = dill.load(file)

    return train_formulas, test_formulas


def create_datasets_spot_simple(dataset_name, set_type, train_size, test_size):
    train_set_dpath, test_set_dpath = create_dataset_directories(dataset_name)
    create_dataset_spot_simple(spot_train(), train_set_dpath, 'train', set_type, train_size)
    create_dataset_spot_simple(spot_test(), test_set_dpath, 'test', set_type, test_size)


def create_dataset_spot_simple(formulas, savepath, set_name, set_type, size):
    filename = f'{set_name}_{set_type}_{size}.pkl'
    with open(os.path.join(savepath, filename), 'wb') as file:
        dill.dump(formulas[0:size], file)
    human_readable_filename = f'{set_name}_{set_type}_{size}.txt'
    with open(os.path.join(savepath, human_readable_filename), 'w') as file:
        for idx, formula in enumerate(formulas[0:size]):
            file.write(f"{idx}: {str(formula)}\n")


def spot_train():
    # a: desk_a; b: desk_b; c: bookshelf; j: juice bottle
    return [
        ('and', ('until', ('not', 'a'), 'c'), ('until', 'True', 'a')),  # !a U c & Fa
        ('and', ('until', ('not', 'b'), 'j'), ('until', 'True', 'b'))  # !b U j & Fb
    ]


def spot_test():
    return [
        # ('and', ('until', ('not', 'a'), 'c'), ('until', 'True', 'a')),
        # ('and', ('until', ('not', 'b'), 'j'), ('until', 'True', 'b')),
        ('until', 'True', ('and', 'c', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'j', ('until', 'True', 'a'))))))),  # F(c & F(a & F(j & Fa)))
        ('until', 'True', ('and', 'c', ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'j', ('until', 'True', 'b'))))))),  # F(c & F(b & F(j & Fb)))

        ('until', 'True', ('and', 'c', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'j', ('until', 'True', 'b'))))))),
        ('until', 'True', ('and', 'c', ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'j', ('until', 'True', 'a'))))))),
        ('until', 'True', ('and', 'j', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'c', ('until', 'True', 'a'))))))),
        ('until', 'True', ('and', 'j', ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'c', ('until', 'True', 'b'))))))),
        ('until', 'True', ('and', 'j', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'c', ('until', 'True', 'b'))))))),
        ('until', 'True', ('and', 'j', ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'c', ('until', 'True', 'a'))))))),

        ('and', ('until', ('not', 'a'), 'c'), ('and', ('until', ('not', 'j'), 'a'), ('and', ('until', ('not', 'b'), 'j'), ('until', 'True', 'b')))),
        ('and', ('until', ('not', 'b'), 'j'), ('and', ('until', ('not', 'c'), 'b'), ('and', ('until', ('not', 'a'), 'c'), ('until', 'True', 'a')))),

    ]


if __name__ == '__main__':
    env_name = "spot"
    create_datasets(save_dpath="..", env_name=env_name, props=ENV2PROPS[env_name], set_types=["soft_strict"], duplicate_ok=False)
