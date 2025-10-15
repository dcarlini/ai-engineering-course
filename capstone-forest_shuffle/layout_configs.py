# layout_configs.py

# Example: Horizontal split into two equal halves
HORIZONTAL_SPLIT = [
    {"x_ratio": 0, "y_ratio": 0, "width_ratio": 0.5, "height_ratio": 1.0},
    {"x_ratio": 0.5, "y_ratio": 0, "width_ratio": 0.5, "height_ratio": 1.0}
]

# Example: Vertical split into two equal halves
VERTICAL_SPLIT = [
    {"x_ratio": 0, "y_ratio": 0, "width_ratio": 1.0, "height_ratio": 0.5},
    {"x_ratio": 0, "y_ratio": 0.5, "width_ratio": 1.0, "height_ratio": 0.5}
]

# Example: Five cards in a cross pattern.
# The coordinates are ratios of the total image dimensions.
# The generator creates a 2W x 2H canvas where W and H are standard card dimensions.
FIVE_CARD_CROSS = [
    # Center card
    {"x_ratio": 0.25, "y_ratio": 0.25, "width_ratio": 0.5, "height_ratio": 0.5},
    # Top card
    {"x_ratio": 0.25, "y_ratio": 0, "width_ratio": 0.5, "height_ratio": 0.25},
    # Bottom card
    {"x_ratio": 0.25, "y_ratio": 0.75, "width_ratio": 0.5, "height_ratio": 0.25},
    # Left card
    {"x_ratio": 0, "y_ratio": 0.25, "width_ratio": 0.25, "height_ratio": 0.5},
    # Right card
    {"x_ratio": 0.75, "y_ratio": 0.25, "width_ratio": 0.25, "height_ratio": 0.5}
]

# Dictionary to easily access layouts by name
PREDEFINED_LAYOUTS = {
    "horizontal_split": HORIZONTAL_SPLIT,
    "vertical_split": VERTICAL_SPLIT,
    "five_card_cross": FIVE_CARD_CROSS,
}
