#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def conjunctions(clauses):
    '''Combines all the clauses separated with 'and's. Note that this codebase only allows for
    binary conjunctions, and not a list of conjunctions as in the PUnS codebase'''

    if len(clauses) == 2:
        return ('and', clauses[0], clauses[1])
    else:
        return ('and', clauses[0], conjunctions(clauses[1::]))

def eventually(prop):
    return ('until','True', prop)

def hard_order(p1,p2):
    return ('until',('not',p2), p1)

def soft_order(seq):
    if len(seq) == 1:
        return ('until', 'True', seq[0])
    else:
        return ('until', 'True', ('and', seq[0], soft_order(seq[1::])))

def sample_waypoints(props):
    a=1

def sample_sequences(visit_waypoints):
    a=1
    
def seq2orders(sequences):
    a=1
    
def orders2clauses(orders, order_type = 'mixed'):
    a=1

def seq2clauses(sequences):
    a=1

def sample_formula():
    a=1
