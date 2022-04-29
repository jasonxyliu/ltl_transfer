#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:47:34 2022

@author: ajshah
"""

from dataset_creator import read_train_test_formulas
from zero_shot_transfer import *
import dill

RESULT_DPATH = '../results'


class Record:

    def __init__(self, train_type = 'hard', train_size = 50, test_type = 'hard', edge_matcher = 'strict', map_id = 0, n_tasks=100):
        self.train_type = train_type
        self.test_type = test_type
        self.train_size = train_size
        self.edge_matcher = edge_matcher
        self.map_id = map_id
        self.n_tasks = n_tasks
        self.data = self.read_all_records()



    def read_all_records(self):

        train_type = self.train_type
        test_type = self.test_type
        train_size = self.train_size
        map_id = self.map_id
        n_tasks = self.n_tasks
        edge_matcher = self.edge_matcher
        train_tasks, test_tasks = read_train_test_formulas(train_set_type = train_type, train_size = train_size, test_set_type = test_type)

        records = []
        result_dpath = os.path.join(RESULT_DPATH, f'{train_type}_{train_size}_{test_type}_{edge_matcher}', f'map_{map_id}')
        filenames = [os.path.join(result_dpath, f'test_ltl_{i}.pkl') for i in range(n_tasks)]

        for (i,f) in enumerate(filenames):
            if os.path.exists(f):
                with open(f,'rb') as file:
                    records.append(dill.load(file))
            else:
                records.append({'transfer_task':test_tasks[i], 'success': 0.0, 'run2sol':defaultdict(list), 'run2traj': {}, 'run2exitcode': 'timeout', 'runtime': 0 })
        return records

    @property
    def success(self):
        return [r['success'] for r in self.data]

    @property
    def runtimes(self):
        return [r['runtimes'] for r in self.data if r['run2exitcode'] != 'timeout']

    @property
    def specification_failure_rate(self):
        num_times  = np.max([len(r['run2exitcode']) for r in records])
        total = len(self.data)
        spec_fails = 0
        for r in self.data:
            if type(r['run2exitcodes']) == dict:
                inc = len([k for k in r['run2exitcodes'] if r['run2exitcodes'][k] == 'specification_fail'])
                spec_fails += inc

def get_results(train_type='hard', edge_matcher = 'relaxed', test_types = None, map_id = 0, train_sizes = [50]):
    
    if not test_types:
        test_types = ['hard','soft','soft_strict','mixed','no_orders']
    results = {}
    for test_type in test_types:
        for train_size in train_sizes:
            results[(train_type, train_size, test_type, edge_matcher)] = Record(train_type, train_size, test_type, edge_matcher, map_id = 0)
    return results



if __name__ == '__main__':

    #TODO: Make this commandline argparse
    #results = get_results('mixed','relaxed')
    results = get_results('mixed','relaxed',['mixed'], train_sizes = [10,20,30,40,50])
    