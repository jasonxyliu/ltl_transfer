#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:29:47 2022
"""
import os
import dill
import numpy as np
from formula_sampler import sample_formula


NAME2PROPS = {
    "minecraft": ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 's'),
    "spot": ('a', 'b', 'c', 'd', 'j', 'p')
}
SET_TYPES = ('hard', 'soft', 'soft_strict', 'no_orders', 'mixed')
TRAIN_SIZES = (10, 20, 30, 40, 50)
TEST_SIZE = 100


def create_datasets(dataset_name, set_types=SET_TYPES, duplicate_ok=True, train_sizes=TRAIN_SIZES, test_size=TEST_SIZE):
    train_set_dpath, test_set_dpath = create_dataset_directories(dataset_name)
    for typ in set_types:
        create_dataset(NAME2PROPS[dataset_name], train_set_dpath, 'train', typ, duplicate_ok, train_sizes)
    for typ in set_types:
        create_dataset(NAME2PROPS[dataset_name], test_set_dpath, 'test', typ, duplicate_ok, [test_size])


def create_dataset_directories(dataset_name):
    dataset_dpath = os.path.join("../experiments/datasets", dataset_name)
    train_set_dpath = os.path.join(dataset_dpath, 'training')
    test_set_dpath = os.path.join(dataset_dpath, 'test')
    if not os.path.exists(dataset_dpath):
        os.mkdir(dataset_dpath)
        os.mkdir(train_set_dpath)
        os.mkdir(test_set_dpath)
    return train_set_dpath, test_set_dpath


def create_dataset(props, savepath, set_name='train', set_type='mixed', duplicate_ok=True, sizes=TRAIN_SIZES):
    n_formulas = np.max(sizes)
    if duplicate_ok:
        formulas = sample_dataset_formulas(props=props, set_type=set_type, n=n_formulas)
    else:
        formulas = sample_dataset_unique_formulas(props=props, set_type=set_type, n=n_formulas)

    for size in sizes:
        filename = f'{set_name}_{set_type}_{size}.pkl'
        with open(os.path.join(savepath, filename), 'wb') as file:
            dill.dump(formulas[0:size], file)
        human_readable_filename = f'{set_name}_{set_type}_{size}.txt'
        with open(os.path.join(savepath, human_readable_filename), 'w') as file:
            for idx, formula in enumerate(formulas[0:size]):
                file.write(f"{idx}: {str(formula)}\n")


def sample_dataset_formulas(props, set_type='mixed', n=50):
    """
    Allowed to sample duplicated LTL formulas into the dataset
    """
    if set_type == 'no_orders':
        formulas = [sample_formula(props=props, orders=False)[0] for _ in range(n)]
    else:
        formulas = [sample_formula(props=props, orders=True, order_type=set_type)[0] for _ in range(n)]
    return formulas


def sample_dataset_unique_formulas(props, set_type='mixed', n=50):
    """
    Only sample unique LTL formulas into the dataset
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


def read_train_test_formulas(dataset_name, train_set_type='mixed', test_set_type='hard', train_size=50, test_size=100):
    dataset_dpath = os.path.join("../experiments/datasets", dataset_name)
    train_set_dpath = os.path.join(dataset_dpath, 'training')
    test_set_dpath = os.path.join(dataset_dpath, 'test')

    train_set_name = f'train_{train_set_type}_{train_size}.pkl'
    test_set_name = f'test_{test_set_type}_{test_size}.pkl'

    with open(os.path.join(train_set_dpath, train_set_name), 'rb') as file:
        train_formulas = dill.load(file)
    with open(os.path.join(test_set_dpath, test_set_name), 'rb') as file:
        test_formulas = dill.load(file)

    return train_formulas, test_formulas


if __name__ == '__main__':
    create_datasets(dataset_name="spot", set_types=["soft_strict"], duplicate_ok=False)
    # filter_datasets(dataset_name="spot", set_types=["soft_strict"], filters=[])
