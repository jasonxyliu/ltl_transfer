#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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

def soft_order_strict(seq):
    if len(seq)==1:
        return ('until','True', seq[0])
    else:
        return ('until','True',('and', seq(0), ('next', soft_order_strict(seq[1::]))))

def hard_order(p1,p2):
    return ('until',('not',p1),p2)


def sample_waypoints(props):
    visit_waypoints = []
    for p in props:
        if np.random.binomial(1,0.5):
            visit_waypoints.append(p)
    return visit_waypoints

def sample_sequences(visit_waypoints):
    
    sequences = []
    new_seq = [visit_waypoints[0]]
        
    for w in visit_waypoints[1::]:
        if np.random.binomian(1,0.5):
            #start a new sequence
            sequences.append(new_seq)
            new_sequence = [w]
        else:
            #continue old sequence
            new_sequence.append(w)
    sequences.append(new_sequence) #add the final sequence
    return sequences
    
def seq2orders(sequences):
    #return binary orders from each sequence
    orders = []
    for seq in sequences:
        for (i,p) in enumerate(seq):
            orders.extend([[p,p2] for p2 in seq[i+1::]])
    return orders


def order2clause(p1,p2, order_type):
    if order_type == 'hard':
        return hard_order(p1,p2)
    elif order_type == 'soft':
        return soft_order([p1,p2])
    else:
        return soft_order_strict([p1,p2])

def orders2clauses(orders, order_type = 'mixed'):
    clauses = []
    for order in orders:
        if order_type == 'mixed':
            if np.random.binomial(1,0.5):
                typ = 'hard'
            else:
                if np.random.binomial(1,0.5):
                    typ = 'soft'
                else:
                    typ = 'soft_strict'
        else:
            typ = order_type
    clauses.append(order2clause(order[0], order[1], typ))
    return clauses
        

def seq2clauses(sequences):
    a=1

def sample_formula():
    a=1
