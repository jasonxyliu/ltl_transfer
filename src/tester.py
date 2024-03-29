import sympy
from zero_shot_transfer import _is_subset_eq


def test_is_subset_eq():
    # test edge, training edges, is_match
    test_samples = (
        ("a", [sympy.simplify_logic("a", form='dnf')], True),
        ("a", [sympy.simplify_logic("~a", form='dnf')], False),
        ("a", [sympy.simplify_logic("b", form='dnf')], False),
        ("a", [sympy.simplify_logic("~b", form='dnf')], False),
        ("!a", [sympy.simplify_logic("~a", form='dnf')], True),
        ("!a", [sympy.simplify_logic("a", form='dnf')], False),
        ("!a", [sympy.simplify_logic("b", form='dnf')], False),
        ("!a", [sympy.simplify_logic("~b", form='dnf')], False),

        ("a", [sympy.simplify_logic("a&b", form='dnf')], True),  # place cup vs. place cup on the plate
        ("!a", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("b", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("!b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("c", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("!c", [sympy.simplify_logic("a&b", form='dnf')], False),

        ("a", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("!a", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("!b", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("c", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("!c", [sympy.simplify_logic("a&~b", form='dnf')], False),

        ("!a", [sympy.simplify_logic("~a&~b", form='dnf')], True),
        ("a", [sympy.simplify_logic("~a&~b", form='dnf')], False),
        ("!b", [sympy.simplify_logic("~a&~b", form='dnf')], True),
        ("b", [sympy.simplify_logic("~a&~b", form='dnf')], False),
        ("!c", [sympy.simplify_logic("~a&~b", form='dnf')], False),
        ("c", [sympy.simplify_logic("~a&~b", form='dnf')], False),

        ("a&b", [sympy.simplify_logic("a", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("~a", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("b", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("~b", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("c", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("~c", form='dnf')], False),

        ("a&!b", [sympy.simplify_logic("a", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("~a", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("~b", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("b", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("c", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("~c", form='dnf')], False),

        ("a&b", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("!a&b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("a&c", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("a&!c", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("!a&c", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("!a&!c", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("c&b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("!c&b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("c&!b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("!c&!b", [sympy.simplify_logic("a&b", form='dnf')], False),

        ("a&!b", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("a&b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("a&c", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("a&!c", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("!a&b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("!a&!b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("c&b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("!c&b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("c&!b", [sympy.simplify_logic("a&~b", form='dnf')], False),
        ("c&!b", [sympy.simplify_logic("a&~b", form='dnf')], False),

        # ("a&b&!c", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("a", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("b", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("!c", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("a&b", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("a&!c", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        # ("b&!c", [sympy.simplify_logic("a&b&~c", form='dnf')], True),
        #
        # ("a&b&!c", [sympy.simplify_logic("a&b&~c&~d", form='dnf')], True),
        # ("a&!b&!c", [sympy.simplify_logic("a&~b", form='dnf')], False),
        # ("a&b&!c&!d", [sympy.simplify_logic("a&~c", form='dnf')], False),

        ("a", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("c", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!c", [sympy.simplify_logic("a|b", form='dnf')], False),

        ("!a", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("a", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("b", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("!b", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("c", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("!c", [sympy.simplify_logic("~a|b", form='dnf')], False),

        ("a|b", [sympy.simplify_logic("a", form='dnf')], True),
        ("a|b", [sympy.simplify_logic("~a", form='dnf')], False),
        ("a|b", [sympy.simplify_logic("b", form='dnf')], True),
        ("a|b", [sympy.simplify_logic("~b", form='dnf')], False),
        ("a|b", [sympy.simplify_logic("c", form='dnf')], False),
        ("a|b", [sympy.simplify_logic("~c", form='dnf')], False),

        ("!a|b", [sympy.simplify_logic("~a", form='dnf')], True),
        ("!a|b", [sympy.simplify_logic("a", form='dnf')], False),
        ("!a|b", [sympy.simplify_logic("b", form='dnf')], True),
        ("!a|b", [sympy.simplify_logic("~b", form='dnf')], False),
        ("!a|b", [sympy.simplify_logic("c", form='dnf')], False),
        ("!a|b", [sympy.simplify_logic("~c", form='dnf')], False),

        ("a|b", [sympy.simplify_logic("a|b", form='dnf')], True),
        ("a|!b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a|!b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("a|c", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("a|!c", [sympy.simplify_logic("a|b", form='dnf')], False),

        ("!a|b", [sympy.simplify_logic("~a|b", form='dnf')], True),
        ("!a|!b", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("!a|c", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("!a|!c", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("a|b", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("c|b", [sympy.simplify_logic("~a|b", form='dnf')], False),
        ("!c|b", [sympy.simplify_logic("~a|b", form='dnf')], False),

        ("!a|!b", [sympy.simplify_logic("~a|~b", form='dnf')], True),
        ("a|!b", [sympy.simplify_logic("~a|~b", form='dnf')], False),
        ("a|b", [sympy.simplify_logic("~a|~b", form='dnf')], False),
        ("a|c", [sympy.simplify_logic("~a|~b", form='dnf')], False),
        ("a|!c", [sympy.simplify_logic("~a|~b", form='dnf')], False),

        ("a&b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("a&c", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("a&!c", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a&b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a&!b", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a&c", [sympy.simplify_logic("a|b", form='dnf')], False),
        ("!a&!c", [sympy.simplify_logic("a|b", form='dnf')], False),

        ("a&b", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("a&!b", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("a&c", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("a&!c", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("!a&b", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("!a&!b", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("!a&c", [sympy.simplify_logic("a|~b", form='dnf')], False),
        ("!a&!c", [sympy.simplify_logic("a|~b", form='dnf')], False),

        ("a|b", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("!a|b", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("!a|!b", [sympy.simplify_logic("a&b", form='dnf')], False),
        ("a|c", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("a|!c", [sympy.simplify_logic("a&b", form='dnf')], True),

        ("a|!b", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("a|b", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("!a|!b", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("a|!c", [sympy.simplify_logic("a&~b", form='dnf')], True),
        ("a|c", [sympy.simplify_logic("a&~b", form='dnf')], True),

        ("a", [sympy.simplify_logic("a&b|c", form='dnf')], False),
        ("a&b", [sympy.simplify_logic("a&b|c", form='dnf')], False),
        ("a&b|c", [sympy.simplify_logic("a|c", form='dnf')], False),
        ("a&b|c", [sympy.simplify_logic("a&b", form='dnf')], True),
        ("a|c", [sympy.simplify_logic("a&b|c", form='dnf')], True),

        ("!a&b|c", [sympy.simplify_logic("~a&b|c", form='dnf')], True),
        ("!a&b|c", [sympy.simplify_logic("c", form='dnf')], True),
        ("!a&b|c", [sympy.simplify_logic("~a&b", form='dnf')], True),
        ("!a&b|c", [sympy.simplify_logic("~a", form='dnf')], False),
        ("!a&b|c", [sympy.simplify_logic("b", form='dnf')], False),
        ("(!a&b)|(c&d)", [sympy.simplify_logic("(~a&b)|c", form='dnf')], False),
        ("(!a&b)|(c&d)", [sympy.simplify_logic("(~a&b)|(c&d)", form='dnf')], True),
        ("(!a&b)|(c&d)", [sympy.simplify_logic("(~a&b)|(c&d&e)", form='dnf')], True),
        ("(!a&b)|c", [sympy.simplify_logic("(~a&b)|c|d", form='dnf')], False),
        ("(!a&b)|c|d", [sympy.simplify_logic("(~a&b)|(c&d)", form='dnf')], True),
    )

    for test_edge, training_edges, truth in test_samples:
        test_edge = sympy.simplify_logic(test_edge.replace('!', '~'), form='dnf')
        pred = _is_subset_eq(test_edge, training_edges[0])
        assert pred == truth, "\ntest egde: %s\ntraining edges: %s\n" \
                              "predicted: %s; ground truth: %s" % (test_edge, training_edges, pred, truth)


if __name__ == "__main__":
    test_is_subset_eq()
