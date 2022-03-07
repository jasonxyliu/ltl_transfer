import sympy
from zero_shot_transfer import match_edges


def test_match_edges():
    test_samples = (
        ("a", [sympy.simplify("a")], True),  # test edge, training edges, is_match
        ("a", [sympy.simplify("~a")], False),
        ("a", [sympy.simplify("b")], False),
        ("a", [sympy.simplify("~b")], False),
        ("!a", [sympy.simplify("~a")], True),
        ("!a", [sympy.simplify("a")], False),
        ("!a", [sympy.simplify("b")], False),
        ("!a", [sympy.simplify("~b")], False),

        ("a", [sympy.simplify("a&b")], True),  # place cup vs. place cup on the plate
        ("!a", [sympy.simplify("a&b")], False),
        ("b", [sympy.simplify("a&b")], True),
        ("!b", [sympy.simplify("a&b")], False),
        ("c", [sympy.simplify("a&b")], False),
        ("!c", [sympy.simplify("a&b")], False),

        ("a", [sympy.simplify("a&~b")], True),
        ("!a", [sympy.simplify("a&~b")], False),
        ("!b", [sympy.simplify("a&~b")], True),
        ("b", [sympy.simplify("a&~b")], False),
        ("c", [sympy.simplify("a&~b")], False),
        ("!c", [sympy.simplify("a&~b")], False),

        ("!a", [sympy.simplify("~a&~b")], True),
        ("a", [sympy.simplify("~a&~b")], False),
        ("!b", [sympy.simplify("~a&~b")], True),
        ("b", [sympy.simplify("~a&~b")], False),
        ("!c", [sympy.simplify("~a&~b")], False),
        ("c", [sympy.simplify("~a&~b")], False),

        ("a&b", [sympy.simplify("a")], False),
        ("a&b", [sympy.simplify("~a")], False),
        ("a&b", [sympy.simplify("b")], False),
        ("a&b", [sympy.simplify("~b")], False),
        ("a&b", [sympy.simplify("c")], False),
        ("a&b", [sympy.simplify("~c")], False),

        ("a&!b", [sympy.simplify("a")], False),
        ("a&!b", [sympy.simplify("~a")], False),
        ("a&!b", [sympy.simplify("~b")], False),
        ("a&!b", [sympy.simplify("b")], False),
        ("a&!b", [sympy.simplify("c")], False),
        ("a&!b", [sympy.simplify("~c")], False),

        ("a&b", [sympy.simplify("a&b")], True),
        ("!a&b", [sympy.simplify("a&b")], False),
        ("a&!b", [sympy.simplify("a&b")], False),
        ("a&c", [sympy.simplify("a&b")], False),
        ("a&!c", [sympy.simplify("a&b")], False),
        ("!a&c", [sympy.simplify("a&b")], False),
        ("!a&!c", [sympy.simplify("a&b")], False),
        ("c&b", [sympy.simplify("a&b")], False),
        ("!c&b", [sympy.simplify("a&b")], False),
        ("c&!b", [sympy.simplify("a&b")], False),
        ("!c&!b", [sympy.simplify("a&b")], False),

        ("a&!b", [sympy.simplify("a&~b")], True),
        ("a&b", [sympy.simplify("a&~b")], False),
        ("a&c", [sympy.simplify("a&~b")], False),
        ("a&!c", [sympy.simplify("a&~b")], False),
        ("!a&b", [sympy.simplify("a&~b")], False),
        ("!a&!b", [sympy.simplify("a&~b")], False),
        ("c&b", [sympy.simplify("a&~b")], False),
        ("!c&b", [sympy.simplify("a&~b")], False),
        ("c&!b", [sympy.simplify("a&~b")], False),
        ("c&!b", [sympy.simplify("a&~b")], False),

        # ("a&b&!c", [sympy.simplify("a&b&~c")], True),
        # ("a", [sympy.simplify("a&b&~c")], True),
        # ("b", [sympy.simplify("a&b&~c")], True),
        # ("!c", [sympy.simplify("a&b&~c")], True),
        # ("a&b", [sympy.simplify("a&b&~c")], True),
        # ("a&!c", [sympy.simplify("a&b&~c")], True),
        # ("b&!c", [sympy.simplify("a&b&~c")], True),
        #
        # ("a&b&!c", [sympy.simplify("a&b&~c&~d")], True),
        # ("a&!b&!c", [sympy.simplify("a&~b")], False),
        # ("a&b&!c&!d", [sympy.simplify("a&~c")], False),

        ("a", [sympy.simplify("a|b")], False),
        ("!a", [sympy.simplify("a|b")], False),
        ("b", [sympy.simplify("a|b")], False),
        ("!b", [sympy.simplify("a|b")], False),
        ("c", [sympy.simplify("a|b")], False),
        ("!c", [sympy.simplify("a|b")], False),

        ("!a", [sympy.simplify("~a|b")], False),
        ("a", [sympy.simplify("~a|b")], False),
        ("b", [sympy.simplify("~a|b")], False),
        ("!b", [sympy.simplify("~a|b")], False),
        ("c", [sympy.simplify("~a|b")], False),
        ("!c", [sympy.simplify("~a|b")], False),

        ("a|b", [sympy.simplify("a")], True),
        ("a|b", [sympy.simplify("~a")], False),
        ("a|b", [sympy.simplify("b")], True),
        ("a|b", [sympy.simplify("~b")], False),
        ("a|b", [sympy.simplify("c")], False),
        ("a|b", [sympy.simplify("~c")], False),

        ("!a|b", [sympy.simplify("~a")], True),
        ("!a|b", [sympy.simplify("a")], False),
        ("!a|b", [sympy.simplify("b")], True),
        ("!a|b", [sympy.simplify("~b")], False),
        ("!a|b", [sympy.simplify("c")], False),
        ("!a|b", [sympy.simplify("~c")], False),

        ("a|b", [sympy.simplify("a|b")], True),
        ("a|!b", [sympy.simplify("a|b")], False),
        ("!a|!b", [sympy.simplify("a|b")], False),
        ("a|c", [sympy.simplify("a|b")], False),
        ("a|!c", [sympy.simplify("a|b")], False),

        ("!a|b", [sympy.simplify("~a|b")], True),
        ("!a|!b", [sympy.simplify("~a|b")], False),
        ("!a|c", [sympy.simplify("~a|b")], False),
        ("!a|!c", [sympy.simplify("~a|b")], False),
        ("a|b", [sympy.simplify("~a|b")], False),
        ("c|b", [sympy.simplify("~a|b")], False),
        ("!c|b", [sympy.simplify("~a|b")], False),

        ("!a|!b", [sympy.simplify("~a|~b")], True),
        ("a|!b", [sympy.simplify("~a|~b")], False),
        ("a|b", [sympy.simplify("~a|~b")], False),
        ("a|c", [sympy.simplify("~a|~b")], False),
        ("a|!c", [sympy.simplify("~a|~b")], False),

        ("a&b", [sympy.simplify("a|b")], False),
        ("a&!b", [sympy.simplify("a|b")], False),
        ("a&c", [sympy.simplify("a|b")], False),
        ("a&!c", [sympy.simplify("a|b")], False),
        ("!a&b", [sympy.simplify("a|b")], False),
        ("!a&!b", [sympy.simplify("a|b")], False),
        ("!a&c", [sympy.simplify("a|b")], False),
        ("!a&!c", [sympy.simplify("a|b")], False),

        ("a&b", [sympy.simplify("a|~b")], False),
        ("a&!b", [sympy.simplify("a|~b")], False),
        ("a&c", [sympy.simplify("a|~b")], False),
        ("a&!c", [sympy.simplify("a|~b")], False),
        ("!a&b", [sympy.simplify("a|~b")], False),
        ("!a&!b", [sympy.simplify("a|~b")], False),
        ("!a&c", [sympy.simplify("a|~b")], False),
        ("!a&!c", [sympy.simplify("a|~b")], False),

        ("a|b", [sympy.simplify("a&b")], True),
        ("!a|b", [sympy.simplify("a&b")], True),
        ("!a|!b", [sympy.simplify("a&b")], False),
        ("a|c", [sympy.simplify("a&b")], True),
        ("a|!c", [sympy.simplify("a&b")], True),

        ("a|!b", [sympy.simplify("a&~b")], True),
        ("a|b", [sympy.simplify("a&~b")], True),
        ("!a|!b", [sympy.simplify("a&~b")], True),
        ("a|!c", [sympy.simplify("a&~b")], True),
        ("a|c", [sympy.simplify("a&~b")], True),

        ("a", [sympy.simplify("a&b|c")], False),
        ("a&b", [sympy.simplify("a&b|c")], False),
        ("a&b|c", [sympy.simplify("a|c")], False),
        ("a&b|c", [sympy.simplify("a&b")], True),
        ("a|c", [sympy.simplify("a&b|c")], True),

        ("!a&b|c", [sympy.simplify("~a&b|c")], True),
        ("!a&b|c", [sympy.simplify("~a&b"), sympy.simplify("c")], True),
        ("!a&b|c", [sympy.simplify("~a"), sympy.simplify("b"), sympy.simplify("c")], True),
        ("!a&b|c", [sympy.simplify("~a"), sympy.simplify("b")], False),
    )

    for test_edge, training_edges, truth in test_samples:
        print("\n")
        pred = match_edges(test_edge, training_edges)
        assert pred == truth, "\ntest egde: %s\ntraining edges: %s\n" \
                              "predicted: %s; ground truth: %s" % (test_edge, training_edges, pred, truth)


if __name__ == "__main__":
    test_match_edges()
