#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:09:39 2022

@author: ajshah
"""

from dataset_creator import *
from ltl_progression import *

dataset_types = ['hard','soft','no_orders','soft_strict','mixed']
training_set_sizes = [10,20,30,40,50]

def create_progression_set(train_set, naive = False, verbose = False):
    n_prog_set = 0
    prog_set = set()
    for (i,f) in enumerate(train_set):
        if verbose:
            print(f'Compiling formula: {i}')
        dfa = get_dfa(f)
        prog_set = prog_set | set(dfa[2].keys())
        if naive:
            n_prog_set = n_prog_set + len(dfa[2])
        else:
            n_prog_set = len(prog_set)
    
    if naive:
        return prog_set, n_prog_set
    else:
        return prog_set

def report_prog_sets(verbose = True):
    prog_sets = {}
    naive_sizes = {}
    for dtype in dataset_types:
        for size in training_set_sizes:
            train_set, _ = read_test_train_formulas(train_set_type=dtype, size=size)
            prog_set, naive_set_size = create_progression_set(train_set, naive = True)
            if verbose:
                print(f"Train set: {dtype}, size: {size} progression set size: {len(prog_set)}, naive size: {naive_set_size}")
            prog_sets[(dtype, size)] = prog_set
            naive_sizes[(dtype, size)] = naive_set_size
    return prog_sets, naive_sizes

def estimate_lpopl_success(verbose = True):
    seen_formulas = {}
    success_rate = {}
    for train_type in dataset_types:
        for test_type in dataset_types:
            train_set, test_set = read_test_train_formulas(train_set_type = train_type, test_set_type=test_type, size = 50)
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

if __name__ == '__main__':
    seen_formulas, success_rate = estimate_lpopl_success()
            