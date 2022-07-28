#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:09:39 2022
"""
from collections import defaultdict
from dataset_creator import read_train_test_formulas, SET_TYPES, TRAIN_SIZES, TEST_SIZE
from ltl_progression import get_dfa


def create_progression_set(train_set, naive=False, verbose=False):
    prog_set = set()
    n_prog_set = 0
    for (i, f) in enumerate(train_set):
        if verbose:
            print(f'Compiling formula: {i}')
        dfa = get_dfa(f)
        prog_set = prog_set | set(dfa[2].keys())
        if naive:  # allow duplicates
            n_prog_set = n_prog_set + len(dfa[2])
        else:
            n_prog_set = len(prog_set)
    
    if naive:
        return prog_set, n_prog_set
    else:
        return prog_set


def report_prog_sets(dataset_name, verbose=True):
    prog_sets = {}
    naive_sizes = {}
    for dtype in SET_TYPES:
        for size in TRAIN_SIZES:
            train_set, _ = read_train_test_formulas(dataset_name=dataset_name, train_set_type=dtype, train_size=size)
            prog_set, naive_set_size = create_progression_set(train_set, naive=True)
            if verbose:
                print(f"Train set: {dtype}, size: {size} progression set size: {len(prog_set)}, naive size: {naive_set_size}")
            prog_sets[(dtype, size)] = prog_set
            naive_sizes[(dtype, size)] = naive_set_size
    return prog_sets, naive_sizes


def estimate_lpopl_success(dataset_name, verbose=True):
    seen_formulas = {}
    success_rate = {}
    for train_type in SET_TYPES:
        for test_type in SET_TYPES:
            train_set, test_set = read_train_test_formulas(dataset_name, train_type, test_type, 50)
            prog_set = create_progression_set(train_set)
            seen = []
            for f in test_set:
                if f in prog_set:
                    seen.append(f)
            seen_formulas[(train_type, test_type)] = seen
            success_rate[(train_type, test_type)] = len(seen)/len(test_set)
            if verbose:
                print(f'LPOPL success rate: {success_rate[(train_type, test_type)]}, train set: {train_type}, test_set: {test_type}')
    return seen_formulas, success_rate


def examine_train_test_sets(dataset_name, train_type, test_type, train_sizes=TRAIN_SIZES):
    test_tasks = None
    for train_size in train_sizes:
        # Unique formulas in training set
        train_tasks, test_tasks = read_train_test_formulas(dataset_name, train_type, test_type, train_size)
        count_unique_formulas(train_tasks, "%s_train_%d contains" % (train_type, train_size))
        # Training formulas also in test set
        train2occurs = defaultdict(int)
        for train_task in train_tasks:
            if train_task in test_tasks:
                train2occurs[train_task] += 1
        print("%d tasks from %s_train_%d occurred in %s_test_%d more than once\n" % (len(train2occurs), train_type, train_size, test_type, TEST_SIZE))
        # for train_task, occurs in train2occurs.items():
        #     print("train_task occurs in test set %d times\n%s\n" % (occurs, str(train_task)))
    # Unique formulas in test set
    count_unique_formulas(test_tasks, "%s_test_%d contains" % (test_type, TEST_SIZE))


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
    # seen_formulas, success_rate = estimate_lpopl_success(dataset_name="spot")
    examine_train_test_sets(dataset_name="spot", train_type='soft_strict', test_type='soft_strict', train_sizes=[50])  # examine train and test sets
