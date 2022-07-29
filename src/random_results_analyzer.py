#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:07:41 2022

@author: ajshah
"""

from result_analyzer import *
import pandas as pd

TEST_TYPES = ['hard','soft','soft_strict','no_orders','mixed']

def get_random_results(test_types=TEST_TYPES, num_steps = [500], map_ids = [0,1,5,6]):
    results = {}
    for test_type in test_types:
        for nsteps in num_steps:
            for map_id in map_ids:
                results[(test_type, nsteps, map_id)] = RandomRecord(test_type = test_type, nsteps = nsteps, map_id = map_id)
    return results

def get_mixed_50_results():
    return get_results(train_types = ['mixed'], test_types = TEST_TYPES, train_sizes=[50], map_ids = [0,1,5,6])

def create_data_table():
    random_results = get_random_results()
    mixed_results = get_mixed_50_results()
    
    data = {}
    i=0
    
    for k in random_results:
        entry = {}
        entry['Transfer Method'] =f'Random Policy {k[1]} steps'
        entry['Test Type'] = k[0]
        entry['Map ID'] = k[2]
        entry['Success Rate'] = np.mean(random_results[k].success)
        data[i] = entry
        i = i+1
    for k in mixed_results:
        entry = {}
        entry['Transfer Method'] = f'LTL Transfer {k[3]}'
        entry['Test Type'] = k[2]
        entry['Map ID'] = k[4]
        entry['Success Rate']  = np.mean(mixed_results[k].success)
        data[i] = entry
        i = i+1
    
    data = pd.DataFrame.from_dict(data, orient = 'index')
    return data        
        
    
    

if __name__ == '__main__':
    random_results = get_random_results()
    mixed_results = get_mixed_50_results()