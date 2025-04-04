NUM_LANDMARKS = 21
FINGER_JOINTS = {
    "thumb_cmc": (1, 2, 3),
    "thumb_mcp": (2, 3, 4),
    "index_finger_mcp": (5, 6, 7),
    "index_finger_pip": (6, 7, 8),
    "middle_finger_mcp": (9, 10, 11),
    "middle_finger_pip": (10, 11, 12),
    "ring_finger_mcp": (13, 14, 15),
    "ring_finger_pip": (14, 15, 16),
    "pinky_mcp": (17, 18, 19),
    "pinky_pip": (18, 19, 20),
}
LABEL_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
}
NUM_LABELS = len(LABEL_MAPPING)
RANDOM_STATE = 42
SPATIAL_THRESHOLD = 0.2
