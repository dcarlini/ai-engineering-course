# layout_configs.py

# Example: Horizontal split into two equal halves
HORIZONTAL_SPLIT = [
    {"x": 0, "y": 0, "width_ratio": 0.5, "height_ratio": 1.0},
    {"x_ratio": 0.5, "y": 0, "width_ratio": 0.5, "height_ratio": 1.0}
]

# Example: Vertical split into two equal halves
VERTICAL_SPLIT = [
    {"x": 0, "y": 0, "width_ratio": 1.0, "height_ratio": 0.5},
    {"x": 0, "y_ratio": 0.5, "width_ratio": 1.0, "height_ratio": 0.5}
]

# Example: Five cards in a cross pattern (adjust ratios based on actual image)
# This assumes a central card and four surrounding cards.
# The ratios are relative to the full image width/height.
FIVE_CARD_CROSS = [
    # Center card (example: 1/3rd width, 1/3rd height, centered)
    {"x_ratio": 1/3, "y_ratio": 1/3, "width_ratio": 1/3, "height_ratio": 1/3},
    # Top card
    {"x_ratio": 1/3, "y_ratio": 0, "width_ratio": 1/3, "height_ratio": 1/3},
    # Bottom card
    {"x_ratio": 1/3, "y_ratio": 2/3, "width_ratio": 1/3, "height_ratio": 1/3},
    # Left card
    {"x_ratio": 0, "y_ratio": 1/3, "width_ratio": 1/3, "height_ratio": 1/3},
    # Right card
    {"x_ratio": 2/3, "y_ratio": 1/3, "width_ratio": 1/3, "height_ratio": 1/3}
]

# Dictionary to easily access layouts by name
PREDEFINED_LAYOUTS = {
    "horizontal_split": HORIZONTAL_SPLIT,
    "vertical_split": VERTICAL_SPLIT,
    "five_card_cross": FIVE_CARD_CROSS,
}
