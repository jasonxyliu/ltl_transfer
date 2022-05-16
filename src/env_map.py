"""
+x: forward, +y: left
+yaw: counter clockwise
"""


CODE2ROT = {
    0: (0, 0, 0),
    1: (0, 0, -90),
    2: (0, 0, 180),
    3: (0, 0, 90)
}


COORD2LOC = {
    (0, 0): (2, 0),

    (3, 3): (4.9, -1.8),  # start loc



    (3, 2): (4.9, -1.55),

    (3, 1): (4.9, -1.2),

    (4, 1): (3.5, -1.2),

    (5, 1): (1.6, -1.2),

    (6, 1): (0.5, -1.2),  # c




    (4, 3): (3.5, -1.8),

    (5, 3): (1.6, -1.8),  # a



    (3, 4): (4.9, -3.5),

    (2, 4): (5.5, -3.5),

    (1, 4): (6.3, -3.5),  # b





    (3, 4): (4.9, -3.5),
    (3, 5): (4.9, -4),
    (3, 6): (4.9, -5),
    (3, 7): (4.9, -6),
    (3, 8): (4.9, -7),
    (3, 9): (4.9, -8),

    (3, 10): (4.9, -9.3),  # j



    (4, 2): (3.5, -1.55),
    (5, 2): (1.6, -1.55),

}


