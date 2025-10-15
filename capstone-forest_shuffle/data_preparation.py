from email import generator
import os
import requests
import shutil
from sklearn.model_selection import train_test_split
from torchvision.io import read_image, write_jpeg # Assuming torchvision is installed
import sys
import json
import csv
import re
from card_data_manager import CardCollection
from symbol_manager import SymbolCollection
from multi_card_image_generator import MultiCardImageGenerator



class DataPreparation:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.main_data_root = os.path.join(self.base_dir, "data")
        self.raw_data_dir = os.path.join(self.main_data_root, "raw_data")
        self.processed_cards_dir = os.path.join(self.main_data_root, "processed_cards")
        self.datasets_dir = os.path.join(self.main_data_root, "datasets")
        self.urls_file = os.path.join(self.base_dir, "config", "card_sets.csv")
        self.card_names_file = os.path.join(self.base_dir, "config", "card_names.csv")
        self.card_names = {} # To store name for each individual card (filename)
        self.card_layout_configs_file = os.path.join(self.base_dir, "config", "card_data.csv")
        self.card_layout_configs = {}
        self.image_configs = {}

    def _load_card_layout_configs(self):
        print(f"Loading card layout configurations from {self.card_layout_configs_file}..."); sys.stdout.flush()
        if not os.path.exists(self.card_layout_configs_file):
            print(f"Error: {self.card_layout_configs_file} not found. Please create it with card_type,row,col,layout."); sys.stdout.flush()
            return

        with open(self.card_layout_configs_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                
                if len(row) < 4:
                    print(f"Skipping malformed line in card_data.csv: {row}"); sys.stdout.flush()
                    continue
                
                card_type, row_str, col_str, layout = row[:4]
                try:
                    row_idx = int(row_str)
                    col_idx = int(col_str)
                except ValueError:
                    print(f"Skipping line with invalid row/col: {row}"); sys.stdout.flush()
                    continue
                
                self.card_layout_configs[(card_type, row_idx, col_idx)] = layout.lower()
        print("Card layout configurations loaded."); sys.stdout.flush()

    def load_card_names(self):
        print(f"Loading card names from {self.card_names_file}..."); sys.stdout.flush()
        if not os.path.exists(self.card_names_file):
            print(f"Error: {self.card_names_file} not found. Please run generate_card_names_config.py first."); sys.stdout.flush()
            return

        cards_csv_path = os.path.join(self.base_dir, 'config', 'cards_by_name.csv')
        if not os.path.exists(cards_csv_path):
            print(f"Error: Master card data file not found at {cards_csv_path}"); sys.stdout.flush()
            return
        card_collection = CardCollection(cards_csv_path)

        with open(self.card_names_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            try:
                file_name_idx = header.index('file_name')
                name_idx = header.index('name')
                suit_idx = header.index('suit') if 'suit' in header else -1
                expansion_idx = header.index('expansion') if 'expansion' in header else -1
            except ValueError as e:
                print(f"Error: Missing expected column in card_names.csv: {e}"); sys.stdout.flush()
                return

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                
                file_name = row[file_name_idx]
                card_name = row[name_idx]

                if not card_name or card_name.isspace():
                    self.card_names[file_name] = {
                        'name': 'None',
                        'type_1': 'None',
                        'type_2': 'None',
                        'suit': 'None',
                        'expansion': 'None',
                        'individual_labels': []
                    }
                    continue

                suit_override = row[suit_idx] if suit_idx != -1 and len(row) > suit_idx else None
                expansion_override = row[expansion_idx] if expansion_idx != -1 and len(row) > expansion_idx else None

                individual_labels = []
                card_details = None
                final_expansion = None
                final_suit = None
                
                found_cards = card_collection.search_by_name(card_name)
                if found_cards:
                    card_details = found_cards[0]
                    
                    # Handle suit override
                    if suit_override and not suit_override.isspace():
                        final_suit = suit_override
                    else:
                        final_suit = card_details.suit

                    # Handle expansion override
                    if expansion_override and not expansion_override.isspace():
                        final_expansion = expansion_override
                    else:
                        final_expansion = card_details.expansion

                    if card_details.type_1 and card_details.type_1.lower() != 'none':
                        individual_labels.append(card_details.type_1)
                    if card_details.type_2 and card_details.type_2.lower() != 'none':
                        individual_labels.append(card_details.type_2)
                    if final_expansion and final_expansion.lower() != 'none':
                        individual_labels.append(final_expansion.strip())
                else:
                    if expansion_override and not expansion_override.isspace():
                        individual_labels.append(expansion_override.strip())
                    final_suit = suit_override # Use override if card not in master list
                    print(f"Warning: Card '{card_name}' not found in CardCollection. Relying on card_names.csv for some data."); sys.stdout.flush()

                self.card_names[file_name] = {
                    'name': card_name,
                    'type_1': card_details.type_1 if card_details else 'None',
                    'type_2': card_details.type_2 if card_details else 'None',
                    'suit': final_suit if final_suit and not final_suit.isspace() else 'None',
                    'expansion': final_expansion if final_expansion and not final_expansion.isspace() else 'None',
                    'individual_labels': sorted(list(set(individual_labels)))
                }
        print("Card names loaded."); sys.stdout.flush()

    def _cleanup(self):
        print(f"Cleaning up old data directory: {self.main_data_root}"); sys.stdout.flush()
        if os.path.exists(self.main_data_root):
            shutil.rmtree(self.main_data_root)
        print("Cleanup complete."); sys.stdout.flush()

    def download_images(self):
        print("Downloading images..."); sys.stdout.flush()
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        if not os.path.exists(self.urls_file):
            print(f"Error: {self.urls_file} not found. Please create it with URL,rows,cols,split_type."); sys.stdout.flush()
            return

        with open(self.urls_file, "r") as f:
            next(f) # Skip header row
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"Skipping malformed line in urls.csv (expected 3 parts: url,rows,cols): {line}")
                    continue
                
                url, rows_str, cols_str = parts
                filename = os.path.basename(url)
                filepath = os.path.join(self.raw_data_dir, filename)

                try:
                    rows = int(rows_str)
                    cols = int(cols_str)
                except ValueError:
                    print(f"Skipping line with invalid rows/cols: {line}")
                    continue

                self.image_configs[filename] = {
                    "url": url,
                    "rows": rows,
                    "cols": cols
                }

                if os.path.exists(filepath):
                    print(f"File already exists: {filename}, skipping download.")
                    continue
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(filepath, "wb") as f_out:
                        for chunk in response.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                    print(f"Downloaded {filename}")
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {url}: {e}")
        print("Image download complete.")

    

    def _split_grid_into_cards(self, card_type: str, rows: int, cols: int):
        print(f"Splitting {card_type} into a {rows}x{cols} grid...")
        output_dir = os.path.join(self.processed_cards_dir, card_type)
        os.makedirs(output_dir, exist_ok=True)

        img_filename = f"{card_type}.jpg"
        img_path = os.path.join(self.raw_data_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"Error: Source image {img_path} not found for grid split.")
            return

        img = read_image(img_path)
        _, height, width = img.shape

        card_width = width // cols
        card_height = height // rows

        for i in range(rows):
            for j in range(cols):
                layout = self.card_layout_configs.get((card_type, i, j), "whole")
                if layout == "discard":
                    continue

                left = j * card_width
                top = i * card_height
                right = left + card_width
                bottom = top + card_height

                card = img[:, top:bottom, left:right]
                # Save with a temporary name, e.g., card_0_0_temp.jpg
                write_jpeg(card, os.path.join(output_dir, f"{card_type}_{i}_{j}_temp.jpg"))
        print(f"Grid split for {card_type} complete. Cards saved to {output_dir}.")

    def _split_card_halves(self, card_type: str):
        print(f"Applying layout for {card_type} cards based on individual configurations...")
        base_card_dir = os.path.join(self.processed_cards_dir, card_type)
        
        for filename in os.listdir(base_card_dir):
            print(f"Processing file: {filename}")
            if filename.endswith("_temp.jpg"):
                card_path = os.path.join(base_card_dir, filename)
                base_name = filename.replace("_temp.jpg", "") # e.g., card_0_0
                print(f"base_name: {base_name}")
                
                # Extract row and col from base_name (e.g., card_0_0 -> 0, 0)
                parts = base_name.split('_')
                if len(parts) == 3 and parts[0] == card_type:
                    row = int(parts[1])
                    col = int(parts[2])
                else:
                    print(f"Warning: Could not parse row/col from {base_name}. Skipping card.")
                    os.remove(card_path) # Remove unprocessable temp file
                    continue

                layout = self.card_layout_configs.get((card_type, row, col), "whole") # Default to whole if not specified
                
                final_full_card_name = f"{base_name}.jpg"
                final_full_card_path = os.path.join(base_card_dir, final_full_card_name)

                # Rename the temporary full card to its final name
                print(f"base_name: {base_name}, final_full_card_name: {final_full_card_name}")
                os.rename(card_path, final_full_card_path)
                card_path = final_full_card_path # Update card_path to the new name

                # Only proceed with splitting if layout is not 'whole'
                if layout != "whole":
                    card_img = read_image(card_path)
                    _, card_height, card_width = card_img.shape

                    if layout == "vertical" or layout == "all":
                        # Top half
                        card_top = card_img[:, :card_height // 2, :]
                        write_jpeg(card_top, os.path.join(base_card_dir, f"{base_name}_top.jpg"))

                        # Bottom half
                        card_bottom = card_img[:, card_height // 2:, :]
                        write_jpeg(card_bottom, os.path.join(base_card_dir, f"{base_name}_bottom.jpg"))

                    if layout == "horizontal" or layout == "all":
                        # Left half
                        card_left = card_img[:, :, :card_width // 2]
                        write_jpeg(card_left, os.path.join(base_card_dir, f"{base_name}_left.jpg"))

                        # Right half
                        card_right = card_img[:, :, card_width // 2:]
                        write_jpeg(card_right, os.path.join(base_card_dir, f"{base_name}_right.jpg"))
        print(f"Individual card layout application for {card_type} complete.")

    

    def prepare_train_val_data(self, label_attribute: str = 'name'):
        print(f"Preparing training and validation data for attribute: {label_attribute}...")
        
        attribute_datasets_dir = os.path.join(self.datasets_dir, label_attribute)
        train_dir = os.path.join(attribute_datasets_dir, "train")
        val_dir = os.path.join(attribute_datasets_dir, "val")

        if os.path.exists(attribute_datasets_dir):
            shutil.rmtree(attribute_datasets_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        all_image_paths = []
        for card_type_dir_name in os.listdir(self.processed_cards_dir):
            card_type_path = os.path.join(self.processed_cards_dir, card_type_dir_name)
            if os.path.isdir(card_type_path):
                for root, _, files in os.walk(card_type_path):
                    for file in files:
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            all_image_paths.append(os.path.join(root, file))

        unique_labels = sorted(list(set([card_info[label_attribute] for card_info in self.card_names.values() if card_info and card_info.get(label_attribute) and card_info.get(label_attribute) != 'None'])))

        for label_value in unique_labels:
            sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
            if not sanitized_label_value:
                continue
            os.makedirs(os.path.join(train_dir, sanitized_label_value), exist_ok=True)
            os.makedirs(os.path.join(val_dir, sanitized_label_value), exist_ok=True)

        images_by_label = {label: [] for label in unique_labels}
        for image_path in all_image_paths:
            filename = os.path.basename(image_path)
            card_info = self.card_names.get(filename)
            if card_info and card_info.get(label_attribute) and card_info.get(label_attribute) != 'None':
                label_value = card_info[label_attribute]
                if label_value in images_by_label:
                    images_by_label[label_value].append(image_path)

        for label_value, images_for_label in images_by_label.items():
            sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
            if not sanitized_label_value:
                continue

            if len(images_for_label) > 1:
                train_images, val_images = train_test_split(images_for_label, test_size=0.2, random_state=42)
            else:
                train_images = images_for_label
                val_images = []

            for image_path in train_images:
                destination_path = os.path.join(train_dir, sanitized_label_value, os.path.basename(image_path))
                shutil.copy(image_path, destination_path)

            for image_path in val_images:
                destination_path = os.path.join(val_dir, sanitized_label_value, os.path.basename(image_path))
                shutil.copy(image_path, destination_path)

        if label_attribute == 'suit':
            print("Adding suit symbol images to the training set...")
            symbol_collection = SymbolCollection()
            symbols_base_dir = os.path.join(self.base_dir, "config", "symbols")
            suit_symbols = symbol_collection.search_by_group('suit')
            for symbol in suit_symbols:
                if symbol.display_name and symbol.display_name in unique_labels:
                    symbol_image_path = os.path.join(symbols_base_dir, symbol.group, symbol.filename)
                    if os.path.exists(symbol_image_path):
                        sanitized_label_value = "".join(c for c in symbol.display_name if c.isalnum() or c in (' ', '_')).rstrip()
                        destination_path = os.path.join(train_dir, sanitized_label_value, symbol.filename)
                        shutil.copy(symbol_image_path, destination_path)
                    else:
                        print(f"Warning: Suit symbol image not found at {symbol_image_path}. Skipping.")

        for label_value in unique_labels:
            sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
            if not sanitized_label_value:
                continue
            for dir_set in [train_dir, val_dir]:
                class_dir = os.path.join(dir_set, sanitized_label_value)
                if os.path.exists(class_dir) and not os.listdir(class_dir):
                    os.rmdir(class_dir)

        print(f"Training and validation data preparation for attribute '{label_attribute}' complete.")

    def prepare_individual_label_datasets(self):
        print("Preparing individual label datasets...")
        individual_labels_datasets_dir = os.path.join(self.datasets_dir, "individual_labels")
        train_dir = os.path.join(individual_labels_datasets_dir, "train")
        val_dir = os.path.join(individual_labels_datasets_dir, "val")

        if os.path.exists(individual_labels_datasets_dir):
            shutil.rmtree(individual_labels_datasets_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        all_card_image_paths = []
        for card_type_dir_name in os.listdir(self.processed_cards_dir):
            card_type_path = os.path.join(self.processed_cards_dir, card_type_dir_name)
            if os.path.isdir(card_type_path):
                for root, _, files in os.walk(card_type_path):
                    for file in files:
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            all_card_image_paths.append(os.path.join(root, file))
        
        symbol_collection = SymbolCollection()
        unique_labels = set()
        for card_info in self.card_names.values():
            if 'individual_labels' in card_info:
                for label in card_info['individual_labels']:
                    if label and label != 'None':
                        unique_labels.add(label)
        for symbol in symbol_collection.get_all_symbols():
            if symbol.group.lower() != 'suit' and symbol.display_name and symbol.display_name != 'None':
                unique_labels.add(symbol.display_name)
        unique_labels = sorted(list(unique_labels))

        for label_value in unique_labels:
            sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
            if not sanitized_label_value:
                continue
            os.makedirs(os.path.join(train_dir, sanitized_label_value), exist_ok=True)
            os.makedirs(os.path.join(val_dir, sanitized_label_value), exist_ok=True)

        train_images, val_images = train_test_split(all_card_image_paths, test_size=0.2, random_state=42)
        
        for image_path in train_images:
            filename = os.path.basename(image_path)
            card_info = self.card_names.get(filename)
            if card_info and 'individual_labels' in card_info:
                for label_value in card_info['individual_labels']:
                    if label_value and label_value != 'None':
                        sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
                        if sanitized_label_value:
                            destination_path = os.path.join(train_dir, sanitized_label_value, filename)
                            shutil.copy(image_path, destination_path)

        for image_path in val_images:
            filename = os.path.basename(image_path)
            card_info = self.card_names.get(filename)
            if card_info and 'individual_labels' in card_info:
                for label_value in card_info['individual_labels']:
                    if label_value and label_value != 'None':
                        sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
                        if sanitized_label_value:
                            destination_path = os.path.join(val_dir, sanitized_label_value, filename)
                            shutil.copy(image_path, destination_path)

        print("Adding symbol images to the training set...")
        symbols_base_dir = os.path.join(self.base_dir, "config", "symbols")
        for symbol in symbol_collection.get_all_symbols():
            if symbol.group.lower() == 'suit':
                continue
            if symbol.display_name and symbol.display_name != 'None':
                symbol_image_path = os.path.join(symbols_base_dir, symbol.group, symbol.filename)
                if os.path.exists(symbol_image_path):
                    sanitized_label_value = "".join(c for c in symbol.display_name if c.isalnum() or c in (' ', '_')).rstrip()
                    if sanitized_label_value:
                        destination_path = os.path.join(train_dir, sanitized_label_value, symbol.filename)
                        shutil.copy(symbol_image_path, destination_path)
                else:
                    print(f"Warning: Symbol image not found at {symbol_image_path}. Skipping.")

        for label_value in unique_labels:
            sanitized_label_value = "".join(c for c in label_value if c.isalnum() or c in (' ', '_')).rstrip()
            if not sanitized_label_value:
                continue
            for dir_set in [train_dir, val_dir]:
                class_dir = os.path.join(dir_set, sanitized_label_value)
                if os.path.exists(class_dir) and not os.listdir(class_dir):
                    os.rmdir(class_dir)
                    
        print("Individual label datasets preparation complete.")

    def create_manifest(self):
        print("Creating manifest file...")
        manifest_data = []
        
        # Create a quick lookup for file paths
        filepath_map = {}
        for root, _, files in os.walk(self.processed_cards_dir):
            for file in files:
                filepath_map[file] = os.path.join(root, file)

        for filename, labels in self.card_names.items():
            if filename in filepath_map and labels.get('name') != 'None':
                types_list = []
                if labels.get('type_1') and labels['type_1'] != 'None':
                    types_list.append(labels['type_1'])
                if labels.get('type_2') and labels['type_2'] != 'None':
                    types_list.append(labels['type_2'])

                manifest_entry = {
                    'file_path': filepath_map[filename],
                    'labels': {
                        'name': labels.get('name'),
                        'types': types_list,
                        'suit': labels.get('suit'),
                        'expansion': labels.get('expansion')
                    }
                }
                manifest_data.append(manifest_entry)

        manifest_path = os.path.join(self.main_data_root, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
        
        print(f"Manifest file created at {manifest_path}")

    def run_all_preparation(self):
        self._cleanup()
        self.download_images()
        self._load_card_layout_configs()
        self.load_card_names()

        for filename, config in self.image_configs.items():
            card_type = os.path.splitext(filename)[0] # e.g., hCards, vCards, trees
            self._split_grid_into_cards(card_type, config["rows"], config["cols"])
            self._split_card_halves(card_type)
        
        # Generate card_names.csv after all cards are processed
        
        # from generate_card_names_config import generate_card_names_config
        # generate_card_names_config()
        
        self.load_card_names()
        self.create_manifest()

        generator = MultiCardImageGenerator()
        generated_images = generator.generate_five_card_cross_images(num_images=30)
        print(f"Generated {len(generated_images)} five-card cross images.")

if __name__ == '__main__':
    dp = DataPreparation()
    dp.run_all_preparation()
    
    # Prepare training and validation data for each desired attribute
    dp.prepare_train_val_data(label_attribute='name')
    # dp.prepare_train_val_data(label_attribute='type_1')
    # dp.prepare_train_val_data(label_attribute='type_2')
    dp.prepare_train_val_data(label_attribute='suit')
    # dp.prepare_train_val_data(label_attribute='combined_type') # Removed as per new logic

    # Prepare datasets for individual labels (type_1, type_2, suit treated separately)
    dp.prepare_individual_label_datasets()