# the size of the pictures that the models expect
model_input_size = 224

# maps a label to a greek letter
label_to_char = {
    1: "Θ",
    2: "Α",
    3: "Β",
    4: "Τ",
    5: "Ξ",
    6: "Ε",
    7: "Κ",
    8: "Ω",
    9: "Μ",
    10: "Φ",
    11: "Ρ",
    12: "Π",
    13: "Γ",
    14: "Ν",
    15: "Λ",
    16: "Ζ",
    17: "Η",
    18: "Υ",
    19: "Ψ",
    20: "Χ",
    21: "Δ",
    22: "Ο",
    23: "Ι",
    24: "Ϲ",
}


# maps a category id to a char label
category_to_label = {
    7: 1,
    8: 2,
    9: 3,
    14: 4,
    17: 5,
    23: 6,
    33: 7,
    45: 8,
    59: 9,
    77: 10,
    100: 11,
    107: 12,
    111: 13,
    119: 14,
    120: 15,
    144: 16,
    150: 17,
    161: 18,
    169: 19,
    177: 20,
    186: 21,
    201: 22,
    212: 23,
    225: 24,
}


def get_data_by_category(category):
    """
    given a category. it returns the category,label,char tuple
    """
    if isinstance(category, str):
        category = int(category)

    label = category_to_label[category]
    char = label_to_char[label]
    return category, label, char
