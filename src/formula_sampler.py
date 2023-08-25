#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def sample_formula(props, orders=True, order_type='mixed', formula_size=5):
    clauses = []

    # sample waypoints to be visited
    visit_props = sample_waypoints(props, formula_size)  # radnom.choices([str(x) for x in props], k=formula_size)
    clauses.extend([eventually(p) for p in sorted(visit_props)])

    # If orders are allowed, create ordering formula clauses
    if orders:
        sequences = sample_sequences(visit_props)
        clauses.extend(seq2clauses(sequences, order_type))
    else:
        sequences = [[p] for p in sorted(visit_props)]

    if len(clauses) == 0:
        return 'True', sequences
    elif len(clauses) == 1:
        return clauses[0], sequences
    else:
        return conjunctions(clauses), sequences


'''Formula template samplers'''
def sample_waypoints(props, formula_size):
    """
    Randomly sample 'formula_size' propositions.
    """
    visit_waypoints = []
    props = list(np.random.permutation(props))
    props = [str(x) for x in props]
    while not visit_waypoints:
        for p in props:
            if np.random.binomial(1, 0.5):
                visit_waypoints.append(p)
    if len(visit_waypoints) > formula_size:
        visit_waypoints = visit_waypoints[0: formula_size]  # need to clip the dfa size
    return visit_waypoints


def sample_sequences(visit_waypoints):
    """
    Randomly generate sequences of propositions.
    """
    sequences = []
    new_seq = [visit_waypoints[0]]

    for w in visit_waypoints[1::]:
        if np.random.binomial(1, 0.5):
            # start a new sequence
            sequences.append(new_seq)
            new_seq = [w]
        else:
            # continue old sequence
            new_seq.append(w)
    sequences.append(new_seq)  # add the final sequence
    return sequences


'''Formula generators'''
def seq2clauses(sequences, order_type='mixed'):
    clauses = []
    for seq in sequences:
        if len(seq) > 1:
            if order_type == 'mixed':  # if order type is mixed, then each seq can have one type or mixed
                seq_order = np.random.choice(['mixed', 'hard', 'soft', 'soft_strict'])
            else:
                seq_order = order_type

            if seq_order == 'soft':
                clauses.append(soft_order(seq))
            elif seq_order == 'soft_strict':
                clauses.append(soft_order_strict(seq))
            else:  # mixed order
                orders = seq2orders([seq])
                clauses.extend(orders2clauses(orders, seq_order))
    return clauses


def seq2orders(sequences):
    """
    Create binary orders of propositions for each proposition sequence.
    """
    orders = []
    for seq in sequences:
        for idx, prop in enumerate(seq):
            orders.extend([[prop, prop2] for prop2 in seq[idx+1::]])
    return orders


def orders2clauses(orders, order_type='mixed'):
    """
    Turn every binary order of propositions into clause with specified 'order_type'.
    """
    clauses = []
    for order in orders:
        if order_type == 'mixed':
            if np.random.binomial(1, 0.5):
                typ = 'hard'
            else:
                if np.random.binomial(1, 0.5):
                    typ = 'soft'
                else:
                    typ = 'soft_strict'
        else:
            typ = order_type
        clauses.append(order2clause(order[0], order[1], typ))
    return clauses


def order2clause(p1, p2, order_type):
    if order_type == 'hard':
        return hard_order(p1, p2)
    elif order_type == 'soft':
        return soft_order([p1, p2])
    else:
        return soft_order_strict([p1, p2])


def conjunctions(clauses):
    """
    Combines all the clauses separated with 'and's.
    Note that this codebase only allows for binary conjunctions, not a list of conjunctions as in the PUnS codebase.
    """
    if len(clauses) == 2:
        return ('and', clauses[0], clauses[1])
    else:
        return ('and', clauses[0], conjunctions(clauses[1::]))


def eventually(prop):
    return ('until', 'True', prop)


def hard_order(p1, p2):
    return ('until', ('not', p2), p1)


def soft_order(seq):
    if len(seq) == 1:
        return ('until', 'True', seq[0])
    else:
        return ('until', 'True', ('and', seq[0], soft_order(seq[1::])))


def soft_order_strict(seq):
    if len(seq) == 1:
        return ('until', 'True', seq[0])
    else:
        return ('until', 'True', ('and', seq[0], ('next', soft_order_strict(seq[1::]))))


def hard_order(p1, p2):
    return ('until', ('not', p1), p2)
