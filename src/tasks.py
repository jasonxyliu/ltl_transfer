"""
This module returns the co-safe ltl tasks that we used in our experiments.
The set of propositional symbols are {a,b,c,d,e,f,g,h,n,s}:
    a: got_wood
    b: used_toolshed
    c: used_workbench
    d: got_grass
    e: used_factory
    f: got_iron
    g: used_bridge
    h: used_axe
    n: is_night
    s: at_shelter
"""


def get_sequence_of_subtasks():
    # Experiment 1: Sequences of Sub-Tasks (Section 5.2 in paper)
    tasks = []
    tasks.append(_get_sequence('ab'))
    tasks.append(_get_sequence('ac'))
    tasks.append(_get_sequence('de'))
    tasks.append(_get_sequence('db'))
    tasks.append(_get_sequence('fae'))
    tasks.append(_get_sequence('abdc'))
    tasks.append(_get_sequence('acfb'))
    tasks.append(_get_sequence('acfc'))
    tasks.append(_get_sequence('faeg'))
    tasks.append(_get_sequence('acfbh'))
    return tasks


def get_interleaving_subtasks():
    # Experiment 2: Interleaving Sub-Tasks (Section 5.3 in paper)
    tasks = []
    tasks.append(_get_sequence('ab'))
    tasks.append(_get_sequence('ac'))
    tasks.append(_get_sequence('de'))
    tasks.append(_get_sequence('db'))
    tasks.append(('and', _get_sequence('ae'), _get_sequence('fe')))
    tasks.append(('and', _get_sequence('dc'), _get_sequence('abc')))
    tasks.append(('and', _get_sequence('fb'), _get_sequence('acb')))
    tasks.append(('and', _get_sequence('fc'), _get_sequence('ac')))
    tasks.append(('and', _get_sequence('aeg'), _get_sequence('feg')))
    tasks.append(('and', _get_sequence('fbh'), _get_sequence('acbh')))
    return tasks


def get_safety_constraints():
    # Experiment 3: Safety Constraints (Section 5.4 in paper)
    tasks = []
    tasks.append(_get_sequence_night('ab'))
    tasks.append(_get_sequence_night('ac'))
    tasks.append(_get_sequence_night('de'))
    tasks.append(_get_sequence_night('db'))
    tasks.append(('and', _get_sequence_night('ae'), _get_sequence_night('fe')))
    tasks.append(('and', _get_sequence_night('dc'), _get_sequence_night('abc')))
    tasks.append(('and', _get_sequence_night('fb'), _get_sequence_night('acb')))
    tasks.append(('and', _get_sequence_night('fc'), _get_sequence_night('ac')))
    tasks.append(('and', _get_sequence_night('aeg'), _get_sequence_night('feg')))
    tasks.append(('and', _get_sequence_night('fbh'), _get_sequence_night('acbh')))
    return tasks


def get_option(goal):
    return _get_sequence(goal)


def get_option_night(goal):
    return _get_sequence_night(goal)


def _get_sequence(seq):
    if len(seq) == 1:
        return ('until', 'True', seq)
    return ('until', 'True', ('and', seq[0], _get_sequence(seq[1:])))


def _sn():
    # returns formula to stay on the shelter
    return ('or', ('not', 'n'), 's')


def _snp(proposition):
    # adds the special constraint to go to the shelter for a proposition
    return ('or', ('and', ('not', 'n'), proposition), ('and', 's', proposition))


def _get_sequence_night(seq):
    if len(seq) == 1:
        return ('until', _sn(), _snp(seq))
    return ('until', _sn(), ('and', _snp(seq[0]), _get_sequence_night(seq[1:])))


######### The following methods are for transfer learning #########
def get_training_tasks():
    """ Training tasks for the transfer tasks. """
    tasks = [
        _get_sequence('ab'),
        # _get_sequence('ac'),
        # _get_sequence('de'),
        # _get_sequence('db'),
        # _get_sequence('fae'),
        # _get_sequence('abdc'),
        # _get_sequence('acfb'),
        # _get_sequence('acfc'),
        # _get_sequence('faeg'),
        # _get_sequence('acfbh')
    ]
    return tasks


def get_transfer_tasks():
    """ Testing tasks for the transfer tasks. """
    tasks = []
    return tasks
