#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:07:41 2022

@author: ajshah
"""

from result_analyzer import *
import pandas as pd
import json

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

typ2string = {'hard':'Hard','soft':'Soft','no_orders':'No Orders','mixed': 'Mixed', 'soft_strict': 'Strict Soft'}
ltl2action_files = [f'results_Transfer_test_{typ}_100.json' for typ in ['hard','soft','soft-strict','no-orders','mixed']]
typ2string2 = {'hard':'Hard','soft':'Soft','no-orders':'No Orders','mixed': 'Mixed', 'soft-strict': 'Strict Soft'}


def get_lpopl_data2(test_types = None):
    if test_types is None:
        test_types = ['hard','soft','soft_strict','no_orders','mixed']
    data = {}
    data[('hard')] = {'success_rate': 0.05 , 'failure_rate': 0}
    data[('soft')] = {'success_rate': 0.06 , 'failure_rate': 0}
    data[('soft_strict')] = {'success_rate': 0.08 , 'failure_rate': 0}
    data[('no_orders')] = {'success_rate': 0.27 , 'failure_rate': 0}
    data[('mixed')] = {'success_rate': 0.12 , 'failure_rate': 0}
    data = dict((k, data[k]) for k in test_types)
    return data

def create_data_table(random_results, mixed_results, ltl2action_results = None, lpopl_results = None):
    #random_results = get_random_results()
    #mixed_results = get_mixed_50_results()
    
    data = {}
    i=0
    
    for k in random_results:
        entry = {}
        entry['Transfer Method'] =f'Random Policy {k[1]} steps'
        entry['Test Type'] = typ2string[k[0]]
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
        entry['Test Type'] = typ2string[k[2]]
        entry['Map ID'] = k[4]
        entry['Success Rate']  = np.mean(mixed_results[k].success)
        entry['Path Lengths'] = mixed_results[k].mean_successful_path_length
        entry['Specification Violation Rate'] = 0
        data[i] = entry
        i = i+1
    if ltl2action_results is not None:
        for k in ltl2action_results:
            entry = {}
            entry['Transfer Method'] = 'LTL2Action'
            entry['Test Type'] = typ2string2[k['Test Set']]
            entry['Map ID' ] = 0
            entry['Success Rate'] = k['num_successes']/(k['num_successes'] + k['num_incompletes'] + k['num_spec_fails'])
            entry['Path Lengths' ] =0
            entry['Specification Violation Rate'] = k['num_spec_fails']/(k['num_successes'] + k['num_incompletes'] + k['num_spec_fails'])
            data[i] = entry
            i = i+1
    if lpopl_results is not None:
        for k in lpopl_results:
            entry = {}
            entry['Transfer Method'] = 'LPOPL'
            entry['Test Type'] = typ2string[k]
            entry['Map ID'] = 0
            entry['Success Rate'] = lpopl_results[k]['success_rate']
            entry['Path Lengths'] = 0
            entry['Specification Violation Rate'] = 0
            data[i] = entry
            i=i+1
    
    data = pd.DataFrame.from_dict(data, orient = 'index')
    return data        

def plot_random_success_rate():
    random_results = get_random_results(num_steps = [500])
    mixed_results = get_mixed_50_results(map_ids = [0])
    ltl2action_results = get_ltl2action_results()
    lpopl_results = get_lpopl_data2()
    data = create_data_table(random_results, mixed_results, ltl2action_results, lpopl_results)
    data = data.loc[data['Transfer Method']!= 'LTL Transfer Constrained']
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [10,8])
        hue_order = ['LTL Transfer Relaxed','LTL2Action','LPOPL','Random Policy 500 steps']
        g = sns.barplot(data = data, x = 'Test Type', y = 'Success Rate', hue = 'Transfer Method', hue_order = hue_order, ci=None)
        g.legend_.remove()
    plt.savefig('figures/random_success_rate.jpg',dpi=400, bbox_inches='tight')

def plot_path_lengths():
    random_results = get_random_results()
    mixed_results = get_mixed_50_results()
    data = create_data_table(random_results, mixed_results)
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [10,8])
        sns.barplot(data = data, x = 'Test Type', y = 'Path Lengths', hue = 'Transfer Method')
        plt.ylim(0,1000)
    plt.savefig('figures/random_path_lengths.jpg',dpi=400, bbox_inches='tight')
    
def plot_spec_failure_rate():
    random_results = get_random_results(num_steps = [500], test_types = ['hard','mixed'])
    mixed_results = get_mixed_50_results(test_types = ['hard','mixed'])
    ltl2action_data = get_ltl2action_results(test_types = ['hard','mixed'])
    lpopl_data = get_lpopl_data2(test_types = ['hard','mixed'])
    
    data = create_data_table(random_results, mixed_results, ltl2action_data, lpopl_data)
    data = data.loc[data['Transfer Method']!= 'LTL Transfer Constrained']
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [10,8])
        hue_order = ['LTL Transfer Relaxed','LTL2Action','LPOPL','Random Policy 500 steps']
        g = sns.barplot(data = data, x = 'Test Type', y = 'Specification Violation Rate', hue = 'Transfer Method', ci=None, hue_order = hue_order)
        #g = sns.barplot(data = data, x = 'Test Type', y = 'Success Rate', hue = 'Transfer Method', ci=None)
        g.legend_.set_title(None)
    plt.savefig('figures/random_failure_rate.jpg',dpi=400, bbox_inches='tight')




def get_ltl2action_results(test_types = None):
    results = []
    if test_types is None:
        test_types = ['hard','soft','soft-strict','no-orders','mixed']
    ltl2action_files = [f'results_Transfer_test_{typ}_100.json' for typ in test_types]
    files = [os.path.join('..','LTL2ActionResults',f) for f in ltl2action_files]
    for (f,typ) in zip(files, test_types):
        with open(f,'r') as file:
            results.append(json.load(file))
            results[-1]['Test Set'] = typ
    return results
    
        
    
    

if __name__ == '__main__':
    random_results = get_random_results()
    mixed_results = get_mixed_50_results()
    ltl2action_results = get_ltl2action_results()
    lpopl_results = get_lpopl_data2()
    data = create_data_table(random_results, mixed_results, ltl2action_results, lpopl_results)
    plot_random_success_rate()
    plot_path_lengths()
    plot_spec_failure_rate()
    a=1