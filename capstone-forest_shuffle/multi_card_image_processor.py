from PIL import Image
import os
from layout_configs import PREDEFINED_LAYOUTS # New import

class MultiCardImageProcessor:
    def __init__(self):
        pass

    def process_image(self, image_path, layout_name=None):
        """
        Takes an image path, identifies individual card regions based on layout_name,
        and returns a list of cropped PIL Image objects.

        layout_name: Name of a predefined layout (e.g., "horizontal_split", "five_card_cross").
        """
        full_image = Image.open(image_path).convert('RGB')
        width, height = full_image.size
        cropped_cards = []

        if layout_name is None:
            # Default to processing the whole image as a single card if no layout is provided
            cropped_cards.append(full_image)
            return cropped_cards

        if layout_name not in PREDEFINED_LAYOUTS:
            raise ValueError(f"Unknown layout name: {layout_name}. Available layouts: {list(PREDEFINED_LAYOUTS.keys())}")
        
        layout_config = PREDEFINED_LAYOUTS[layout_name]

        for region in layout_config:
            # Calculate absolute pixel coordinates from ratios or use absolute values
            x = int(region.get('x_ratio', 0) * width)
            y = int(region.get('y_ratio', 0) * height)
            w = int(region.get('width_ratio', 1.0) * width)
            h = int(region.get('height_ratio', 1.0) * height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 > x1 and y2 > y1:
                cropped_cards.append(full_image.crop((x1, y1, x2, y2)))
            else:
                print(f"Warning: Invalid region {region} for image {image_path}. Skipping.")

        return cropped_cards

# Example usage (for testing this new file)
if __name__ == '__main__':
    # You would need a sample image that contains multiple cards
    # For now, let's use a dummy image or assume one exists
    dummy_multi_card_image_path = "path/to/your/multi_card_image.jpg" # Replace with actual path

    # Example layout for a 2x2 grid of cards, each 100x150 pixels
    # This example is now defined in layout_configs.py

    if os.path.exists(dummy_multi_card_image_path):
        processor = MultiCardImageProcessor()
        # Use a predefined layout name
        cards = processor.process_image(dummy_multi_card_image_path, layout_name="horizontal_split")
        for i, card_img in enumerate(cards):
            card_img.save(f"cropped_card_{i}.jpg")
            print(f"Saved cropped_card_{i}.jpg")
    else:
        print(f"Dummy multi-card image not found at {dummy_multi_card_image_path}. Please create one for testing.")
