#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:29:47 2022

@author: ajshah
"""

from formula_sampler import *
import dill
import os
import json

set_types = ['hard','soft','soft_strict','no_orders','mixed']
dataset_path = '../datasets'
training_set_path = os.path.join(dataset_path, 'training')
test_set_path = os.path.join(dataset_path, 'test')

def create_datasets(set_types = set_types, train_sizes = [10,20,30,40,50], test_size = 100):
    # Check and create the directories
    create_dataset_directories()
    #Create the training set
    for typ in set_types:
        create_dataset(set_name = 'train', set_type = typ, sizes = train_sizes)
    for typ in set_types:
        create_dataset(set_name = 'test', set_type = typ, sizes = [test_size])


def create_dataset(set_name = 'train', set_type = 'mixed', sizes = [10,20,30,40,50]):
    if set_name == 'train':
        savepath = training_set_path
    else:
        savepath = test_set_path
    
    n_formulas = np.max(sizes)
    formulas = sample_dataset_formulas(set_type = set_type, n = n_formulas)
    
    for size in sizes:
        filename = f'{set_name}_{set_type}_{size}.pkl'
        human_readable_filename = f'{set_name}_{set_type}_{size}.json'
        with open(os.path.join(savepath, filename), 'wb') as file:
            dill.dump(formulas[0:size], file)
        #with open(os.path.join(savepath, human_readable_filename), 'w') as file:
        #    json.dump(formulas[0:size], file, indent = 4)

def read_test_train_formulas(train_set_type = 'mixed', test_set_type = 'mixed', size = 50):
    train_set_name = f'train_{train_set_type}_{size}.pkl'
    test_set_name = f'test_{test_set_type}_100.pkl'
    
    with open(os.path.join(training_set_path, train_set_name), 'rb') as file:
        train_formulas = dill.load(file)
    with open(os.path.join(test_set_path, test_set_name),'rb') as file:
        test_formulas = dill.load(file)
    
    return train_formulas, test_formulas


def create_dataset_directories():
    dataset_path = '../datasets'
    if not os.path.exists(dataset_path): 
        os.mkdir(dataset_path)
        os.mkdir(training_set_path)
        os.mkdir(test_set_path)

def sample_dataset_formulas(set_type = 'mixed', n = 50):
    
    if set_type == 'no_orders':
        formulas = [sample_formula(orders = False)[0] for i in range(n)]
    else:
        formulas = [sample_formula(orders = True, order_type = set_type)[0] for i in range(n)]
    return formulas    


if __name__ == '__main__':
    create_datasets()

    
        