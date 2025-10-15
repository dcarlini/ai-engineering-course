import os
import json
import random
from PIL import Image
from layout_configs import FIVE_CARD_CROSS # Import the specific layout

class MultiCardImageGenerator:
    def __init__(self, manifest_path='data/manifest.json', output_dir='data/generated_multi_cards'):
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.card_categories = self._load_card_image_categories()
        self.standard_card_size = (300, 466) # Based on observed dimensions of full cards

    def _load_card_image_categories(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Please run data_preparation.py first.")
        
        with open(self.manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        card_categories = {
            'center': [], # For tree cards
            'top': [],
            'bottom': [],
            'left': [],
            'right': [],
            'any_non_tree': [] # Fallback for any non-tree card
        }

        for item in manifest_data:
            file_path = item['file_path']
            card_name = item['labels']['name']
            card_types = item['labels'].get('types', [])

            if card_name == 'None':
                continue # Skip items without a valid name

            # Categorize based on card types and file suffixes
            if "Tree" in card_types:
                card_categories['center'].append(item) # Store full item
            elif file_path.endswith('_top.jpg'):
                card_categories['top'].append(item)
            elif file_path.endswith('_bottom.jpg'):
                card_categories['bottom'].append(item)
            elif file_path.endswith('_left.jpg'):
                card_categories['left'].append(item)
            elif file_path.endswith('_right.jpg'):
                card_categories['right'].append(item)
            else:
                # Full cards that are not trees can be used as generic side cards
                if "Tree" not in card_types:
                    card_categories['any_non_tree'].append(item)

        # Ensure categories have enough cards for sampling
        if not card_categories['center']:
            raise ValueError("Not enough 'Tree' cards found in manifest for center position.")
        if len(card_categories['top']) < 1 or len(card_categories['bottom']) < 1 or \
           len(card_categories['left']) < 1 or len(card_categories['right']) < 1:
            print("Warning: Not enough specific split cards for all side positions. Falling back to any_non_tree cards.")
            # Fill up missing specific categories with any_non_tree cards
            for cat in ['top', 'bottom', 'left', 'right']:
                if not card_categories[cat]:
                    card_categories[cat] = card_categories['any_non_tree']
            if len(card_categories['any_non_tree']) < 4:
                raise ValueError("Not enough non-tree cards for side positions, even with fallback.")

        return card_categories

    def generate_image(self, layout_config, output_filename="multi_card_image.jpg"):
        """
        Generates a single composite image based on the given layout configuration,
        with optional side cards, and returns manifest data for the generated image.
        """
        if len(layout_config) != 5:
            raise ValueError("Layout config must define exactly 5 regions for the five_card_cross layout.")

        canvas_width = self.standard_card_size[0] * 2
        canvas_height = self.standard_card_size[1] * 2

        composite_image = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0)) # Black background

        position_map = {
            0: 'center',
            1: 'top',
            2: 'bottom',
            3: 'left',
            4: 'right',
        }

        cards_to_draw_info = [] # List of dictionaries for manifest
        used_card_paths = set()

        # --- Process Center Card (Mandatory) ---
        center_region = layout_config[0]
        available_center_cards = [item for item in self.card_categories['center'] if item['file_path'] not in used_card_paths]
        if not available_center_cards:
            raise ValueError("Not enough distinct 'Tree' cards found in manifest for center position.")
        chosen_center_card = random.choice(available_center_cards)
        cards_to_draw_info.append({
            'position': 'center',
            'card_name': chosen_center_card['labels']['name'],
            'original_file_path': chosen_center_card['file_path']
        })
        used_card_paths.add(chosen_center_card['file_path'])

        # --- Process Side Cards (Optional) ---
        side_positions = [1, 2, 3, 4] # Indices for top, bottom, left, right
        for i in side_positions:
            if random.choice([True, False]): # Randomly decide to include this card
                region = layout_config[i]
                category_key = position_map.get(i)
                
                available_cards = [item for item in self.card_categories[category_key] if item['file_path'] not in used_card_paths]
                if not available_cards:
                    available_cards = [item for item in self.card_categories['any_non_tree'] if item['file_path'] not in used_card_paths]
                if not available_cards:
                    print(f"Warning: Not enough distinct cards for {category_key} position. Skipping.")
                    continue # Skip if no distinct card can be found

                chosen_card = random.choice(available_cards)
                cards_to_draw_info.append({
                    'position': category_key,
                    'card_name': chosen_card['labels']['name'],
                    'original_file_path': chosen_card['file_path']
                })
                used_card_paths.add(chosen_card['file_path'])

        # --- Draw all selected cards and collect manifest data ---
        for card_info_entry in cards_to_draw_info:
            position = card_info_entry['position']
            # Find the corresponding region in layout_config
            region_index = next(i for i, pos in position_map.items() if pos == position)
            region = layout_config[region_index]

            card_image = Image.open(card_info_entry['original_file_path']).convert('RGB')
            print(f"Debug: Original card image dimensions for {position}: {card_image.size}")
            
            target_width = int(region.get('width_ratio', 1.0) * canvas_width)
            target_height = int(region.get('height_ratio', 1.0) * canvas_height)
            print(f"Debug: Target dimensions for {position}: ({target_width}, {target_height})")
            
            card_image = card_image.resize((target_width, target_height))

            x = int(region.get('x_ratio', 0) * canvas_width)
            y = int(region.get('y_ratio', 0) * canvas_height)
            
            composite_image.paste(card_image, (x, y))
        
        output_path = os.path.join(self.output_dir, output_filename)
        composite_image.save(output_path)
        print(f"Generated {output_path}")

        # Return manifest data for this image
        return {
            'file_path': output_path,
            'cards': cards_to_draw_info
        }

    def save_multi_card_manifest(self, manifest_data, filename="multi_card_manifest.json"):
        """
        Saves the collected multi-card manifest data to a JSON file.
        """
        manifest_path = os.path.join(self.output_dir, filename)
        absolute_manifest_path = os.path.abspath(manifest_path)
        print(f"Debug: Resolved absolute path for manifest: {absolute_manifest_path}")
        print(f"Debug: Attempting to save multi-card manifest to: {absolute_manifest_path}")
        try:
            with open(absolute_manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=4)
            print(f"Debug: Multi-card manifest successfully saved to {absolute_manifest_path}")
            print(f"Debug: Contents of output directory ({self.output_dir}) after save: {os.listdir(self.output_dir)}")
        except Exception as e:
            print(f"Error: Failed to save multi-card manifest to {absolute_manifest_path}. Error: {e}")

    def generate_five_card_cross_images(self, num_images=30):
        """
        Generates multiple images with the five_card_cross layout and a manifest.
        """
        generated_paths = []
        multi_card_manifest = []
        for i in range(num_images):
            output_filename = f"five_card_cross_{i+1}.jpg"
            manifest_entry = self.generate_image(FIVE_CARD_CROSS, output_filename)
            generated_paths.append(manifest_entry['file_path'])
            multi_card_manifest.append(manifest_entry)
        
        self.save_multi_card_manifest(multi_card_manifest)
        print(f"Generated {len(generated_paths)} five-card cross images.")
        return generated_paths

if __name__ == '__main__':
    generator = MultiCardImageGenerator()
    generated_images = generator.generate_five_card_cross_images(num_images=30)
    print(f"Generated {len(generated_images)} five-card cross images.")