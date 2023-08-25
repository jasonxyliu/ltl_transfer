#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:47:34 2022
"""
import os
import dill
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import seaborn as sns
from dataset_creator import read_train_test_formulas
from zero_shot_transfer import *

RESULT_DPATH = '../results_test'
RANDOM_RESULT_DPATH = '../random_results'
rc = {'axes.labelsize': 24, 'axes.titlesize': 32, 'legend.fontsize': 18, 'legend.title_fontsize': 18, 'xtick.labelsize': 20, 'ytick.labelsize': 20}


class Record:

    def __init__(self, train_type='hard', train_size=50, test_type='hard', edge_matcher='strict', map_id=0, n_tasks=100):
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
        train_tasks, test_tasks = read_train_test_formulas(train_set_type = train_type, train_size = 50, test_set_type = test_type)
        train_tasks = train_tasks[0:train_size]

        records = []
        result_dpath = os.path.join(RESULT_DPATH, f'{train_type}_{train_size}_{test_type}_{edge_matcher}', f'map_{map_id}')
        filenames = [os.path.join(result_dpath, f'test_ltl_{i}.pkl') for i in range(n_tasks)]

        for i, f in enumerate(filenames):
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    records.append(dill.load(file))
            else:
                records.append({'transfer_task': test_tasks[i], 'success': 0.0, 'run2sol': defaultdict(list), 'run2traj': {}, 'run2exitcode': 'timeout', 'runtime': 0})
        return records

    @property
    def mean_successful_runtimes(self):
        runtimes = []
        for d in self.data:
            if type(d['run2exitcode']) == dict:
                if d['run2exitcode'][0] == 0:
                    runtimes.append(d['precomp_time'])
        return np.mean(runtimes)
    
    @property
    def success(self):
        return [r['success'] for r in self.data]

    @property
    def runtimes(self):
        return [r['runtime'] for r in self.data if r['run2exitcode'] != 'timeout']
    
    @property
    def mean_successful_runtimes(self):
        runtimes = []
        for d in self.data:
            if type(d['run2exitcode']) == dict:
                if d['run2exitcode'][0] == 0:
                    runtimes.append(d['runtime'])
        return np.mean(runtimes)

    @property
    def specification_failure_rate(self):
        num_times = np.max([len(r['run2exitcode']) for r in self.data])
        total = len(self.data)
        spec_fails = 0
        for r in self.data:
            if type(r['run2exitcode']) == dict:
                inc = len([k for k in r['run2exitcode'] if r['run2exitcode'][k] == 'specification_fail'])
                spec_fails += inc
        return spec_fails/(len(self.data)*num_times)
    
    @property
    def mean_successful_path_length(self):
        pathlengths = []
        for d in self.data:
            if type(d['run2exitcode']) == dict:
                for k in d['run2exitcode']:
                    if d['run2exitcode'][k] == 0:
                        flat_traj = [l for sublist in d['run2traj'][k] for l in sublist]
                        pathlengths.append(len(flat_traj))
        return np.mean(pathlengths)

class RandomRecord:

    def __init__(self, train_type='random', nsteps=500, test_type='hard', edge_matcher='relaxed', map_id=0, n_tasks=100):
        self.train_type = train_type
        self.test_type = test_type
        self.nsteps = nsteps
        self.edge_matcher = edge_matcher
        self.map_id = map_id
        self.n_tasks = n_tasks
        self.data = self.read_all_records()

    def read_all_records(self):
        train_type = self.train_type
        test_type = self.test_type
        nsteps = self.nsteps
        map_id = self.map_id
        n_tasks = self.n_tasks
        edge_matcher = self.edge_matcher
        train_tasks, test_tasks = read_train_test_formulas(train_set_type = 'hard', train_size = 50, test_set_type = test_type)
        train_tasks = train_tasks[0:50]

        records = []
        result_dpath = os.path.join(RANDOM_RESULT_DPATH, f'{nsteps}_steps', f'random_{self.test_type}_relaxed', f'map_{map_id}')
        filenames = [os.path.join(result_dpath, f'test_ltl_{i}.pkl') for i in range(n_tasks)]

        for i, f in enumerate(filenames):
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    records.append(dill.load(file))
            else:
                records.append({'transfer_task': test_tasks[i], 'success': 0.0, 'run2sol': defaultdict(list), 'run2traj': {}, 'run2exitcode': 'timeout', 'runtime': 0})
        return records

    @property
    def success(self):
        return [r['success'] for r in self.data]

    @property
    def runtimes(self):
        return [r['runtime'] for r in self.data if r['run2exitcode'] != 'timeout']
    
    @property
    def mean_successful_runtimes(self):
        runtimes = []
        for d in self.data:
            if type(d['run2exitcode']) == dict:
                if d['run2exitcode'][0] == 0:
                    runtimes.append(d['runtime'])
        return np.mean(runtimes)

    @property
    def specification_failure_rate(self):
        num_times = np.max([len(r['run2exitcode']) for r in self.data])
        total = len(self.data)
        spec_fails = 0
        for r in self.data:
            if type(r['run2exitcode']) == dict:
                inc = len([k for k in r['run2exitcode'] if r['run2exitcode'][k] == 'specification_fail'])
                spec_fails += inc
        return spec_fails/(len(self.data)*num_times)
    
    @property
    def mean_successful_path_length(self):
        pathlengths = []
        for d in self.data:
            if type(d['run2exitcode']) == dict:
                for k in d['run2exitcode']:
                    if d['run2exitcode'][k] == 0:
                        #flat_traj = [l for sublist in d['run2traj'][k] for l in sublist]
                        pathlengths.append(len(d['run2traj'][k]))
        return np.mean(pathlengths)


def get_results(train_type='hard', edge_matcher='relaxed', test_types=None, map_ids=[0], train_sizes=[50]):
    if not test_types:
        test_types = ['hard', 'soft', 'soft_strict', 'mixed', 'no_orders']
    results = {}
    for test_type in test_types:
        for train_size in train_sizes:
            for map_id in map_ids:
                results[(train_type, train_size, test_type, edge_matcher, map_id)] = Record(train_type, train_size, test_type, edge_matcher, map_id=map_id)
    return results


def get_success_CI(results, trials=100, CI=0.95):
    success = {k: np.mean(results[k].success) for k in results}
    success_CI = {}
    for k in success:
        successful_tries = trials * success[k]
        failed_tries = trials - successful_tries
        lower_q = (1 - CI)/2
        upper_q = 0.5 + CI/2
        lower = beta.ppf(lower_q, successful_tries+1, failed_tries+1)
        upper = beta.ppf(upper_q, successful_tries+1, failed_tries+1)
        success_CI[k] = (lower, success[k], upper)
    return success_CI


def get_results(train_types, test_types, train_sizes, map_ids):
    results = {}
    for train_type in train_types:
        for test_type in test_types:
            for train_size in train_sizes:
                for map_id in map_ids:
                    record = Record(train_type, train_size, test_type, 'rigid', map_id=map_id)
                    results[(train_type, train_size, test_type, 'rigid', map_id)] = record
                    record = Record(train_type, train_size, test_type, 'relaxed', map_id=map_id)
                    results[(train_type, train_size, test_type, 'relaxed', map_id)] = record
    return results

def get_random_results(test_types, nsteps = [500, 1000]):
    a=1

type_names = {'hard': 'Hard',
              'soft': 'Soft',
              'soft_strict': 'Strict Soft',
              'no_orders': 'No-orders',
              'mixed': 'Mixed'}


def create_data_table(results):
    data = {}
    i = 0
    # dataframe columns 'Train Set', 'Size', 'Test Set', 'Map', 'Transfer Method'

    for k in results:
        data[i] = {}
        data[i]['Train Set'] = type_names[k[0]]
        data[i]['Size'] = k[1]
        data[i]['Test Set'] = type_names[k[2]]
        data[i]['Transfer Method'] = 'General' if k[3] == 'relaxed' else 'Constrained'
        data[i]['Map'] = k[4]
        data[i]['Success Rate'] = np.mean(results[k].success)
        data[i]['Successful'] = np.sum(results[k].success)
        data[i]['Fails'] = len(results[k].success) - data[i]['Successful']
        i = i+1
    data = pd.DataFrame.from_dict(data, orient='index')
    return data


def lpopl_success_rates():
    data = {}
    data[0] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 5, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.02}
    data[1] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 10, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.05}
    data[2] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 15, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.07}
    data[3] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 20, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.08}
    data[4] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 30, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.1}
    data[5] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 40, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.1}
    data[6] = {'Train Set': 'Mixed', 'Test Set': 'Mixed', 'Size': 50, 'Transfer Method': 'LPOPL', 'Map': 0, 'Success Rate': 0.12}
    data = pd.DataFrame.from_dict(data, orient='index')
    return data


def plot_fig2():
    """line plot comparison with LPOPL"""
    train_types = ['mixed']
    test_types = ['mixed']
    train_sizes = [5, 10, 15, 20, 30, 40, 50]
    map_ids = [0]
    CI = 0.95
    data_ours = create_data_table(get_results(train_types, test_types, train_sizes, map_ids))
    data_lpopl = lpopl_success_rates()
    data = pd.concat([data_ours, data_lpopl], axis=0, ignore_index=True)
    
    means = data.groupby(['Size', 'Transfer Method']).mean()
    means_ours = data_ours.groupby(['Size', 'Transfer Method']).mean()
    
    # Decide color palette and assign colors
    palette = sns.color_palette('deep')
    colors = {}
    matchers = list(means.loc[means.index[0][0]].index)
    matchers_ours = list(means_ours.loc[means_ours.index[0][0]].index)
    
    for i, m in enumerate(matchers):
        colors[m] = palette[i % len(palette)]
    
    data = pd.concat([data_ours, data_lpopl], axis=0, ignore_index = True)

    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize=[10, 8])
        # sns.lineplot(data=data, x='Size', y='Success Rate', hue='Transfer Method', style='Transfer Method', dashes=False, markers=True, hue_order=matchers)
        
        # Create the line plots
        sns.lineplot(data=data, x='Size', y='Success Rate', hue='Transfer Method', style='Transfer Method', dashes=False, markers=True, hue_order=matchers)
        
        # Create the error bars
        for m in matchers_ours:
            mean_locs = means['Success Rate'][:, m]
            
            successes = data_ours.groupby(['Size', 'Transfer Method']).sum()['Successful'][:, m]
            fails = data_ours.groupby(['Size', 'Transfer Method']).sum()['Fails'][:, m]
            lo_q = 0.5 - CI/2
            hi_q = 0.5 + CI/2
            lo = mean_locs - np.array([beta.ppf(lo_q, s, f) for (s, f) in zip(successes, fails)])
            hi = np.array([beta.ppf(hi_q, s, f) for (s, f) in zip(successes, fails)]) - mean_locs
            err = np.array(list(zip(lo, hi))).transpose()
            
            plt.errorbar(mean_locs.index, mean_locs, err, fmt='none', ecolor=colors[m], capsize=10, capthick=2)
        plt.ylim(0, 1.1)
        plt.xlim(0, 55)
        plt.xlabel('Training Set Size')
        plt.legend(loc='center right')
        plt.savefig('figures/fig_2.png', dpi=400, bbox_inches='tight')


def plot_fig3A():
    """Line plots for mixed to different training types for rigid edge matching"""
    train_types = ['mixed']
    test_types = ['hard', 'soft', 'soft_strict', 'no_orders', 'mixed']
    train_sizes = [5, 10, 15, 20, 30, 40, 50]
    map_ids = [0,1,5,6]
    CI = 0.95
    data = create_data_table(get_results(train_types, test_types, train_sizes, map_ids))
    
    data = data.loc[data['Transfer Method'] == 'Constrained']
    # data_relaxed = data.loc[data['Transfer Method'] == 'General']
    
    means = data.groupby(['Size', 'Test Set']).mean()
    # means_ours = data_ours.groupby(['Size', 'Test Set']).mean()
    
    # Decide color palette and assign colors
    palette = sns.color_palette('deep')
    colors = {}
    test_types = list(means.loc[means.index[0][0]].index)
    # $matchers_ours = list(means_ours.loc[means_ours.index[0][0]].index)
    
    for i, m in enumerate(test_types):
        colors[m] = palette[i % len(palette)]
    
    # data = pd.concat([data_ours, data_lpopl], axis=0, ignore_index = True)

    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [10, 8])
        # sns.lineplot(data=data, x='Size', y='Success Rate', hue='Transfer Method', style='Transfer Method', dashes=False, markers=True, hue_order=matchers)
        
        # Create the line plots
        sns.lineplot(data=data, x='Size', y='Success Rate', hue='Test Set', style='Test Set', dashes=False, markers=True, hue_order=test_types, ci=None)
        
        # Create the error bars
        for m in test_types:
            mean_locs = means['Success Rate'][:, m]
            
            successes = data.groupby(['Size', 'Test Set']).sum()['Successful'][:, m]
            fails = data.groupby(['Size', 'Test Set']).sum()['Fails'][:, m]
            lo_q = 0.5 - CI/2
            hi_q = 0.5 + CI/2
            lo = mean_locs - np.array([beta.ppf(lo_q, s, f) for (s, f) in zip(successes, fails)])
            hi = np.array([beta.ppf(hi_q, s, f) for (s, f) in zip(successes, fails)]) - mean_locs
            err = np.array(list(zip(lo, hi))).transpose()
            
            plt.errorbar(mean_locs.index, mean_locs, err, fmt='none', ecolor=colors[m], capsize=10, capthick=2)
        plt.ylim(0, 1.1)
        plt.xlim(0, 55)
        plt.xlabel('Training Set Size')
        plt.legend(loc='upper right')
        plt.savefig('figures/fig_3a.jpg', dpi=400, bbox_inches='tight')


def plot_fig3B():
    """Line plots for mixed to set types for relaxed edge matching"""
    train_types = ['mixed']
    test_types = ['hard', 'soft', 'soft_strict', 'no_orders', 'mixed']
    train_sizes = [5, 10, 15, 20, 30, 40, 50]
    map_ids = [0,]
    CI = 0.95
    data = create_data_table(get_results(train_types, test_types, train_sizes, map_ids))
    
    data = data.loc[data['Transfer Method'] == 'General']
    # data_relaxed = data.loc[data['Transfer Method'] == 'General']
    
    means = data.groupby(['Size', 'Test Set']).mean()
    # means_ours = data_ours.groupby(['Size', 'Test Set']).mean()
    
    # Decide color palette and assign colors
    palette = sns.color_palette('deep')
    colors = {}
    test_types = list(means.loc[means.index[0][0]].index)
    # matchers_ours = list(means_ours.loc[means_ours.index[0][0]].index)
    
    for i, m in enumerate(test_types):
        colors[m] = palette[i % len(palette)]
    
    # data = pd.concat([data_ours, data_lpopl], axis=0, ignore_index = True)

    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize = [10, 8])
        # sns.lineplot(data=data, x='Size', y ='Success Rate', hue='Transfer Method', style='Transfer Method', dashes=False, markers=True, hue_order=matchers)
        
        # Create the line plots
        sns.lineplot(data=data, x='Size', y='Success Rate', hue='Test Set', style='Test Set', dashes=False, markers=True, hue_order=test_types)
        
        # Create the error bars
        for m in test_types:
            mean_locs = means['Success Rate'][:, m]

            successes = data.groupby(['Size', 'Test Set']).sum()['Successful'][:, m]
            fails = data.groupby(['Size', 'Test Set']).sum()['Fails'][:, m]
            lo_q = 0.5 - CI/2
            hi_q = 0.5 + CI/2
            lo = mean_locs - np.array([beta.ppf(lo_q, s, f) for (s, f) in zip(successes, fails)])
            hi = np.array([beta.ppf(hi_q, s, f) for (s, f) in zip(successes, fails)]) - mean_locs
            err = np.array(list(zip(lo, hi))).transpose()
            
            plt.errorbar(mean_locs.index, mean_locs, err, fmt='none', ecolor=colors[m], capsize=10, capthick=2)
        plt.ylim(0, 1.1)
        plt.xlim(0, 55)
        plt.xlabel('Training Set Size')
        plt.legend(loc='lower right')
        plt.savefig('figures/fig_3b.jpg', dpi=400, bbox_inches='tight')


def fig_4A():
    """heatmap for cross type rigid"""
    test_types = ['hard', 'soft', 'soft_strict', 'no_orders', 'mixed']
    train_types= test_types
    train_sizes = [50]
    map_ids = [0]
    CI = 0.95
    data = create_data_table(get_results(train_types, test_types, train_sizes, map_ids))
    data = data.loc[data['Transfer Method'] == 'Constrained']
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize=[12, 10])
        table = data.pivot_table(values='Success Rate', index='Train Set', columns='Test Set')
        sns.heatmap(data=table, annot=True, vmin=0, vmax=1)
        plt.savefig('figures/fig_4a.png', dpi=400, bbox_inches='tight')


def fig_4B():
    """heatmap for cross type relaxed"""
    test_types = ['hard', 'soft', 'soft_strict', 'no_orders', 'mixed']
    train_types= test_types
    train_sizes = [50]
    map_ids = [0]
    CI = 0.95
    data = create_data_table(get_results(train_types, test_types, train_sizes, map_ids))
    data = data.loc[data['Transfer Method'] == 'General']
    with sns.plotting_context('poster', rc=rc):
        plt.figure(figsize=[12, 10])
        table = data.pivot_table(values='Success Rate', index='Train Set', columns='Test Set')
        sns.heatmap(data=table, annot=True, vmin=0, vmax=1)
        plt.savefig('figures/fig_4b.png', dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    # TODO: Make this commandline argparse
    # results = get_results('mixed', 'relaxed')
    # results = get_results('mixed', 'relaxed', ['mixed'], train_sizes=[10,20,30,40,50])
    train_types = ['mixed']
    test_types = ['mixed']
    train_sizes = [5, 10, 15, 20, 30, 40, 50]
    map_ids = [0]
    results = get_results(train_types, test_types, train_sizes, map_ids)

    # for train_type in train_types:
    #     for test_type in test_types:
    #         for train_size in train_sizes:
    #             for map_id in map_ids:
    #                 record = Record(train_type, train_size, test_type, 'rigid')
    #                 results[(train_type, train_size, test_type, 'rigid', map_id)] = record
    #                 record = Record(train_type, train_size, test_type, 'relaxed')
    #                 results[(train_type, train_size, test_type, 'relaxed', map_id)] = record

    # get_success_CI = get_success_CI(results)
