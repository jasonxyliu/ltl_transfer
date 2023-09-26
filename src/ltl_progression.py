"""
This module contains functions to progress co-safe LTL formulas such as:
    (
        'and',
        ('until','True', ('and', 'd', ('until','True','c'))),
        ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
    )
The main function is 'get_dfa' which receives a co-safe LTL formula and progresses
it over all possible valuations of the propositions. It returns all possible progressions
of the formula in the form of a DFA.
"""
from sympy import *
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not
import collections


def get_dfa(ltl_formula):
    propositions = extract_propositions(ltl_formula)
    propositions.sort()
    truth_assignments = _get_truth_assignments(propositions)

    # simplify 'ltl_formula' by removing no_orders clause repeated as inner most clause in soft if any
    ltl_formula = _progress(ltl_formula, '')

    # Creating DFA using progression
    ltl2states = {'False': -1, ltl_formula: 0}
    edge2assignments = {}

    visited, queue = set([ltl_formula]), collections.deque([ltl_formula])
    while queue:
        formula = queue.popleft()
        if formula in ['True', 'False']:
            continue
        for truth_assignment in truth_assignments:
            # progressing formula
            f_progressed = _progress(formula, truth_assignment)
            if f_progressed not in ltl2states:
                new_node = len(ltl2states) - 1
                ltl2states[f_progressed] = new_node
            # adding edge
            edge = (ltl2states[formula], ltl2states[f_progressed])
            if edge not in edge2assignments:
                edge2assignments[edge] = []
            edge2assignments[edge].append(truth_assignment)

            if f_progressed not in visited:
                visited.add(f_progressed)
                queue.append(f_progressed)

    # Adding initial and accepting states
    initial_state = 0
    accepting_states = [ltl2states['True']]

    # Reducing edges formulas to its minimal form...
    # NOTE: this might take a while since we are using a very
    #       inefficient python library for logic manipulations
    edges_tuple = []
    for edge, truth_assignments in edge2assignments.items():
        f = _get_formula(truth_assignments, propositions)
        edges_tuple.append((edge[0], edge[1], f))
    # Adding self-loops for 'True' and 'False'
    edges_tuple.append((ltl2states['True'], ltl2states['True'], 'True'))
    edges_tuple.append((ltl2states['False'], ltl2states['False'], 'True'))

    return initial_state, accepting_states, ltl2states, edges_tuple


def extract_propositions(ltl_formula):
    return list(set(_get_propositions(ltl_formula)))


def _get_propositions(ltl_formula):
    if type(ltl_formula) == str:
        if ltl_formula in ['True', 'False']:
            return []
        return [ltl_formula]

    if ltl_formula[0] in ['not', 'next']:
        return _get_propositions(ltl_formula[1])

    # 'and', 'or', 'until'
    return _get_propositions(ltl_formula[1]) + _get_propositions(ltl_formula[2])


def _get_truth_assignments(propositions):
    """
    computing all possible Truth value assignments for propositions,
    represented by a list of true propositions
    e.g. ['a', 'b', 'c'] -> ['', 'a', 'b', 'ab', 'c', 'ac', 'bc', 'abc']
    """
    truth_assignments = []
    for p in range(2**len(propositions)):
        truth_assignment = ""
        p_id = 0
        while p > 0:
            if p % 2 == 1:
                truth_assignment += propositions[p_id]
            p //= 2
            p_id += 1
        truth_assignments.append(truth_assignment)
    return truth_assignments


def _progress(ltl_formula, truth_assignment):
    """
    Implement LTL progression rules
    """
    if type(ltl_formula) == str:
        # proposition, True or False
        if len(ltl_formula) == 1:
            # ltl_formula is a proposition
            if ltl_formula in truth_assignment:
                return 'True'
            else:
                return 'False'
        # ltl_formula is True or False
        return ltl_formula

    if ltl_formula[0] == 'not':
        # negations should be over propositions only according to the cosafe ltl syntactic restriction
        result = _progress(ltl_formula[1], truth_assignment)
        if result == 'True':
            return 'False'
        elif result == 'False':
            return 'True'
        else:
            raise NotImplementedError("The following formula doesn't follow the cosafe syntactic restriction: " + str(ltl_formula))

    if ltl_formula[0] == 'and':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)
        if res1 == 'True' and res2 == 'True': return 'True'
        if res1 == 'False' or res2 == 'False': return 'False'
        if res1 == 'True': return res2
        if res2 == 'True': return res1
        if res1 == res2:   return res1
        if _subsume_until(res1, res2): return res2
        if _subsume_until(res2, res1): return res1
        return ('and', res1, res2)

    if ltl_formula[0] == 'or':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)
        if res1 == 'True'  or res2 == 'True'  : return 'True'
        if res1 == 'False' and res2 == 'False': return 'False'
        if res1 == 'False': return res2
        if res2 == 'False': return res1
        if res1 == res2:    return res1
        if _subsume_until(res1, res2): return res1
        if _subsume_until(res2, res1): return res2
        return ('or', res1, res2)

    if ltl_formula[0] == 'next':
        return ltl_formula[1]

    if ltl_formula[0] == 'until':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)

        if res1 == 'False':
            f1 = 'False'
        elif res1 == 'True':
            f1 = ('until', ltl_formula[1], ltl_formula[2])
        else:
            f1 = ('and', res1, ('until', ltl_formula[1], ltl_formula[2]))

        if res2 == 'True':
            return 'True'
        if res2 == 'False':
            return f1
        return res2  # ('or', res2, f1)


def _subsume_until(f1, f2):
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'until':
            f2 = f2[2]
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False


def _is_prop_formula(f):
    # returns True if the formula does not contains temporal operators
    return 'next' not in str(f) and 'until' not in str(f)


def _get_formula(truth_assignments, propositions):
    """
    e.g. ['ab', 'abc'], 'abc' -> (a & b & ~c) | (a & b & c) -> a & b
    """
    dnfs = []
    props = dict([(p, symbols(p)) for p in propositions])
    for truth_assignment in truth_assignments:
        dnf = []
        for p in props:
            if p in truth_assignment:
                dnf.append(props[p])
            else:
                dnf.append(Not(props[p]))
        dnfs.append(And(*dnf))
    formula = Or(*dnfs)
    formula = simplify_logic(formula, form='dnf')
    formula = str(formula).replace('(', '').replace(')', '').replace('~', '!').replace(' ', '')
    return formula


if __name__ == "__main__":
    # Test if output DFA has self-edge (0, 0)
    # DFA state and progressed state should be equal if LTLs are simplified, but complete simplify_logic not implemented
    # ltl_formula = ('and', ('until','True', 'a'), ('and', ('until', 'True', 'b'), ('and', ('until', 'True', 'c'), ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'b', ('until', 'True', 'c'))))))))
    # ltl_formula = ('and', ('until', 'True', 'a'), ('and', ('until', 'True', 'c'), ('and', ('until', 'True', 'd'), ('and', ('until', 'True', 's'), ('until', 'True', ('and', 'c', ('until', 'True', 's')))))))
    # ltl_formula = ('and', ('until', 'True', 's'), ('until', 'True', ('and', 'c', ('until', 'True', 's'))))

    # initial_state, accepting_states, ltl2state, edges = get_dfa(ltl_formula)
    # print(initial_state)
    # print(accepting_states)
    # for ltl, state in ltl2state.items():
    #     print(state, ltl)
    # for edge in edges:
    #     print(edge)

    # Test manually defined test tasks for Spot demos
    test_tasks = [
        ('until', 'True', 'a'),  # Fa
        ('and', ('until', 'True', 'a'), ('until', 'True', 'b')),  # Fa & Fb
        ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'c')),  # Fa & Fb & Fc
        ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 's')),  # Fa & Fb & Fs
        ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'k')),  # Fa & Fb & Fk
        ('and', ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'c')), ('until', 'True', 'd')),  # Fa & Fb & Fc & Fd
        ('and', ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'c')), ('until', 'True', 's')),  # Fa & Fb & Fc & Fs
        ('and', ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'c')), ('until', 'True', 'k')),  # Fa & Fb & Fc & Fk
        ('and', ('and', ('and', ('and', ('until', 'True', 'a'), ('until', 'True', 'b')), ('until', 'True', 'c')), ('until', 'True', 'k')), ('until', 'True', 's')),  # Fa & Fb & Fc & Fk & Fs

        ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'c', ('until', 'True', 'd'))))))),  # F(b & F(a & F(c & Fd))))
        ('until', 'True', ('and', 's', ('until', 'True', 'a'))),  # F(s & Fa): fetch and deliver
        ('until', 'True', ('and', 's', ('until', 'True', 'b'))),  # F(s & Fb): fetch and deliver
        ('until', 'True', ('and', 'a', ('until', 'True', 'b'))),  # F(a & Fb)
        ('until', 'True', ('and', 'b', ('until', 'True', 'a'))),  # F(b & Fa)
        ('until', 'True', ('and', 'a', ('until', 'True', ('and', 's', ('until', 'True', 'c'))))),  # F(a & F(s & Fc)): fetch and deliver
        ('until', 'True', ('and', 'b', ('until', 'True', ('and', 's', ('until', 'True', 'c'))))),  # F(b & F(s & Fc)): fetch and deliver
        ('until', 'True', ('and', 's', ('until', 'True', ('and', 'a', ('until', 'True', 'c'))))),  # F(s & F(a & Fc))
        ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'b', ('until', 'True', 'c'))))),  # F(a & F(b & Fc))
        ('until', 'True', ('and', 's', ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'k', ('until', 'True', 'a'))))))),  # F(s & F(a & F(k & Fa)))): fetch and deliver
        ('until', 'True', ('and', 'a', ('until', 'True', ('and', 'b', ('until', 'True', ('and', 'c', ('until', 'True', 'd'))))))),  # F(a & F(b & F(c & Fd))))

        ('until', 'True', ('and', 's', ('next', ('until', 'True', 'a')))),  # F(s & XFa): fetch and deliver
        ('until', 'True', ('and', 'b', ('next', ('until', 'True', 's')))),  # F(b & XFs)
        ('until', 'True', ('and', 'a', ('next', ('until', 'True', 'b')))),  # F(a & XFb)
        ('until', 'True', ('and', 'b', ('next', ('until', 'True', 'a')))),  # F(b & XFa)
        ('until', 'True', ('and', 'a', ('next', ('until', 'True', ('and', 'b', ('next', ('until', 'True', 'c'))))))),  # F(a & XF(b & XFc))
        ('until', 'True', ('and', 'a', ('next', ('until', 'True', ('and', 's', ('next', ('until', 'True', 'b'))))))),  # F(a & XF(s & XFb)): fetch and deliver
        ('until', 'True', ('and', 'b', ('next', ('until', 'True', ('and', 's', ('next', ('until', 'True', 'a'))))))),  # F(b & XF(s & XFa)): fetch and deliver
        ('until', 'True', ('and', 's', ('next', ('until', 'True', ('and', 'b', ('next', ('until', 'True', 'a'))))))),  # F(s & XF(b & XFa))
        ('until', 'True', ('and', 'k', ('next', ('until', 'True', 'b')))),  # F(k & XFb): fetch and deliver
        ('until', 'True', ('and', 'k', ('next', ('until', 'True', 'a')))),  # F(k & XFa): fetch and deliver

        ('and', ('until', ('not', 'a'), 's'), ('until', 'True', 'a')),  # !a U s & Fa: fetch and deliver
        ('and', ('until', ('not', 'b'), 'a'), ('until', 'True', 'b')),  # !b U a & Fb
        ('and', ('until', ('not', 'a'), 'b'), ('until', 'True', 'a')),  # !a U b & Fa
        ('and', ('and', ('until', ('not', 'b'), 'a'), ('until', ('not', 'c'), 'b')), ('until', 'True', 'c')),  # !b U a & !c U b & Fc
        ('and', ('until', ('not', 'b'), 'k'), ('until', 'True', 'b')),  # !b U k & Fb: fetch and deliver
        ('and', ('until', ('not', 'b'), 'c'), ('until', 'True', 'b')),  # !b U c & Fb
        ('and', ('and', ('until', ('not', 'a'), 's'), ('until', ('not', 'b'), 'a')), ('until', 'True', 'b')),  # !a U s & !b U a & Fb
        ('and', ('and', ('until', ('not', 's'), 'a'), ('until', ('not', 'b'), 's')), ('until', 'True', 'b')),  # !s U a & !b U s & Fb
        ('and', ('and', ('until', ('not', 'b'), 'a'), ('until', ('not', 's'), 'b')), ('until', 'True', 's')),  # !b U a & !s U b & Fs
        ('and', ('and', ('until', ('not', 'a'), 'b'), ('until', ('not', 's'), 'a')), ('until', 'True', 's')),  # !a U b & !s U a & Fs

        ('and', ('until', 'True', 'a'), ('until', 'True', ('and', 'b', ('until', 'True', 'c')))),  # Fa & F(b & Fc)
        ('and', ('until', 'True', 'a'), ('and', ('until', ('not', 'c'), 'b'), ('until', 'True', 'c'))),  # Fa & !c U b & Fc
        ('and', ('and', ('until', 'True', ('and', 'a', ('until', 'True', 'b'))), ('until', ('not', 'c'), 'a')), ('until', 'True', 'c')), # F(a & Fb) & !c U a & Fc
        ('and', ('until', 'True', 'a'), ('until', 'True', ('and', 'b', ('next', ('until', 'True', 'c'))))),  # Fa & F(b & XFc)
        ('and', ('and', ('until', 'True', ('and', 'a', ('until', 'True', 'b'))), ('until', ('not', 'c'), 'b')), ('until', 'True', 'c')), # F(a & Fb) & !c U b & Fc
        ('and', ('and', ('until', 'True', ('and', 'b', ('until', 'True', 'a'))), ('until', ('not', 'c'), 'b')), ('until', 'True', 'c')), # F(b & Fa) & !c U b & Fc
        ('and', ('until', 'True', 'c'), ('and', ('until', ('not', 's'), 'a'), ('until', 'True', 's'))),  # Fc & (!s U a & Fs)
        ('and', ('until', 'True', 'c'), ('and', ('until', ('not', 'a'), 's'), ('until', 'True', 'a'))),  # Fc & (!a U s & Fa)
        ('and', ('until', 'True', 'c'), ('and', ('until', ('not', 's'), 'b'), ('until', 'True', 's'))),  # Fc & (!s U b & Fs)
        ('and', ('until', 'True', 'c'), ('and', ('until', ('not', 'b'), 's'), ('until', 'True', 'b'))),  # Fc & (!b U s & Fb)
    ]

    for ltl_formula in test_tasks:
        initial_state, accepting_states, ltl2state, edges = get_dfa(ltl_formula)
        print(f"LTL: {ltl_formula}")
        print(initial_state)
        print(accepting_states)
        for ltl, state in ltl2state.items():
            print(state, ltl)
        for edge in edges:
            print(edge)
        print()
