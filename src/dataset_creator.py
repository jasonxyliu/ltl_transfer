#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:29:47 2022

@author: ajshah
"""
import dill
import os
import json
from collections import defaultdict
import numpy as np
from formula_sampler import sample_formula


SET_TYPES = ('hard', 'soft', 'soft_strict', 'no_orders', 'mixed')
TRAIN_SIZES = (10, 20, 30, 40, 50)
TEST_SIZE = 100
DATASET_DPATH = '../datasets'
TRAIN_SET_DPATH = os.path.join(DATASET_DPATH, 'training')
TEST_SET_DPATH = os.path.join(DATASET_DPATH, 'test')


def create_datasets(set_types=SET_TYPES, duplicate_ok=True, train_sizes=TRAIN_SIZES, test_size=TEST_SIZE):
    create_dataset_directories()
    for typ in set_types:
        create_dataset(set_name='train', set_type=typ, duplicate_ok=duplicate_ok, sizes=train_sizes)
    for typ in set_types:
        create_dataset(set_name='test', set_type=typ, duplicate_ok=duplicate_ok, sizes=[test_size])


def create_dataset_directories():
    if not os.path.exists(DATASET_DPATH):
        os.mkdir(DATASET_DPATH)
        os.mkdir(TRAIN_SET_DPATH)
        os.mkdir(TEST_SET_DPATH)


def create_dataset(set_name='train', set_type='mixed', duplicate_ok=True, sizes=TRAIN_SIZES):
    if set_name == 'train':
        savepath = TRAIN_SET_DPATH
    else:
        savepath = TEST_SET_DPATH

    n_formulas = np.max(sizes)
    if duplicate_ok:
        formulas = sample_dataset_formulas(set_type=set_type, n=n_formulas)
    else:
        formulas = sample_dataset_unique_formulas(set_type=set_type, n=n_formulas)

    for size in sizes:
        filename = f'{set_name}_{set_type}_{size}.pkl'
        with open(os.path.join(savepath, filename), 'wb') as file:
            dill.dump(formulas[0:size], file)
        # human_readable_filename = f'{set_name}_{set_type}_{size}.json'
        # with open(os.path.join(savepath, human_readable_filename), 'w') as file:
        #     json.dump(formulas[0:size], file, indent=4)


def sample_dataset_formulas(set_type='mixed', n=50):
    """
    Allowed to sample duplicated LTL formulas into the dataset
    """
    if set_type == 'no_orders':
        formulas = [sample_formula(orders=False)[0] for _ in range(n)]
    else:
        formulas = [sample_formula(orders=True, order_type=set_type)[0] for _ in range(n)]
    return formulas


def sample_dataset_unique_formulas(set_type='mixed', n=50):
    """
    Only sample unique LTL formulas into the dataset
    """
    formulas = []
    num_samples_sofar = 0
    while num_samples_sofar < n:
        if set_type == 'no_orders':
            formula = sample_formula(orders=False)[0]
        else:
            formula = sample_formula(orders=True, order_type=set_type)[0]
        if formula not in formulas:
            formulas.append(formula)
            num_samples_sofar += 1
    return formulas


def read_train_test_formulas(train_set_type='mixed', test_set_type='mixed', train_size=50):
    train_set_name = f'train_{train_set_type}_{train_size}.pkl'
    test_set_name = f'test_{test_set_type}_100.pkl'

    with open(os.path.join(TRAIN_SET_DPATH, train_set_name), 'rb') as file:
        train_formulas = dill.load(file)
    with open(os.path.join(TEST_SET_DPATH, test_set_name), 'rb') as file:
        test_formulas = dill.load(file)

    return train_formulas, test_formulas


def examine_train_test_sets(train_type, test_type, train_sizes=TRAIN_SIZES):
    test_tasks = None
    for train_size in train_sizes:
        # Unique formulas in training set
        train_tasks, test_tasks = read_train_test_formulas(train_type, test_type, train_size)
        print_prompt = "%s_train_%d contains" % (train_type, train_size)
        count_unique_formulas(train_tasks, print_prompt)
        # Duplicated training formulas in both train and test sets
        train2occurs = defaultdict(int)
        for train_task in train_tasks:
            if train_task in test_tasks:
                train2occurs[train_task] += 1
        print("%d tasks from train_%d occurred in test_%d more than once\n" % (len(train2occurs), train_size, TEST_SIZE))
        # for train_task, occurs in train2occurs.items():
        #     print("train_task occurs in test set %d times\n%s\n" % (occurs, str(train_task)))
    # Unique formulas in test set
    print_prompt = "%s_test_%d contains" % (test_type, TEST_SIZE)
    count_unique_formulas(test_tasks, print_prompt)


def count_unique_formulas(tasks, print_prompt):
    task2occurs = defaultdict(int)
    for task in tasks:
        task2occurs[task] += 1

    unique_tasks = 0
    for task, occurs in task2occurs.items():
        if occurs > 1:
            print("task: %s\noccurances %d" % (str(task), occurs))
        unique_tasks += 1
    print(print_prompt + ": %d unique tasks" % unique_tasks)


if __name__ == '__main__':
    create_datasets(set_types=["no_orders"], duplicate_ok=False)
    examine_train_test_sets(train_type='no_orders', test_type='no_orders')  # examine train and test sets
