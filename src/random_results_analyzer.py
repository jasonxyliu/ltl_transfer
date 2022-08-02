#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:07:41 2022

@author: ajshah
"""

from result_analyzer import *
import pandas as pd

TEST_TYPES = ['hard','soft','soft_strict','no_orders','mixed']

def get_random_results(test_types=TEST_TYPES, num_steps = [500, 1000], map_ids = [0,1,5,6]):
    results = {}
    for test_type in test_types:
        for nsteps in num_steps:
            for map_id in map_ids:
                results[(test_type, nsteps, map_id)] = RandomRecord(test_type = test_type, nsteps = nsteps, map_id = map_id)
    return results

def get_mixed_50_results(test_types = TEST_TYPES, train_sizes = [50], map_ids = [0,1,5,6]):
    return get_results(train_types = ['mixed'], test_types = test_types, train_sizes=train_sizes, map_ids = map_ids)

def create_data_table(random_results, mixed_results):
    #random_results = get_random_results()
    #mixed_results = get_mixed_50_results()
    
    data = {}
    i=0
    
    for k in random_results:
        entry = {}
        entry['Transfer Method'] =f'Random Policy {k[1]} steps'
        entry['Test Type'] = k[0]
        entry['Map ID'] = k[2]
        entry['Success Rate'] = np.mean(random_results[k].success)
        entry['Path Lengths'] = random_results[k].mean_successful_path_length
        entry['Specification Violation Rate'] = random_results[k].specification_failure_rate
        data[i] = entry
        i = i+1
    for k in mixed_results:
        entry = {}
        edge_match = 'Constrained' if k[3] == 'rigid' else 'Relaxed'
        entry['Transfer Method'] = f'LTL Transfer {edge_match}'
        entry['Test Type'] = k[2]
        entry['Map ID'] = k[4]
        entry['Success Rate']  = np.mean(mixed_results[k].success)
        entry['Path Lengths'] = mixed_results[k].mean_successful_path_length
        entry['Specification Violation Rate'] = 0
        data[i] = entry
        i = i+1
    
    data = pd.DataFrame.from_dict(data, orient = 'index')
    return data        

def plot_random_success_rate():
    random_results = get_random_results()
    mixed_results = get_mixed_50_results(map_ids = [0])
    data = create_data_table(random_results, mixed_results)
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [12,8])
        sns.barplot(data = data, x = 'Test Type', y = 'Success Rate', hue = 'Transfer Method')
    plt.savefig('figures/random_success_rate.jpg',dpi=400, bbox_inches='tight')

def plot_path_lengths():
    random_results = get_random_results()
    mixed_results = get_mixed_50_results()
    data = create_data_table(random_results, mixed_results)
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [12,8])
        sns.barplot(data = data, x = 'Test Type', y = 'Path Lengths', hue = 'Transfer Method')
        plt.ylim(0,1000)
    plt.savefig('figures/random_path_lengths.jpg',dpi=400, bbox_inches='tight')
    
def plot_spec_failure_rate():
    random_results = get_random_results(test_types = ['hard','mixed'])
    mixed_results = get_mixed_50_results(test_types = ['hard','mixed'])
    data = create_data_table(random_results, mixed_results)
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [12,8])
        sns.barplot(data = data, x = 'Test Type', y = 'Specification Violation Rate', hue = 'Transfer Method')
    plt.savefig('figures/random_failure_rate.jpg',dpi=400, bbox_inches='tight')
        
    
    

if __name__ == '__main__':
    #data = create_data_table()
    plot_random_success_rate()
    plot_path_lengths()
    plot_spec_failure_rate()