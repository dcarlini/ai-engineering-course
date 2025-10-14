import os
import csv
import easyocr
from PIL import Image
import numpy as np
from card_data_manager import CardCollection
import cv2
from rapidfuzz import process, fuzz

import re

def _normalize_text(text):
    """
    Normalizes text by converting to lowercase and removing non-alphanumeric characters,
    keeping spaces and apostrophes.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove all non-alphanumeric characters except spaces and apostrophes
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text

def _combine_ocr_results(ocr_results, x_threshold=50):
    """
    Combines consecutive OCR results into phrases based on their bounding boxes.
    """
    combined_phrases = []
    if not ocr_results:
        return combined_phrases

    current_phrase_text = ""
    current_phrase_bbox = None

    for (bbox, text, prob) in ocr_results:
        # Extract coordinates for current text
        x_min_curr = bbox[0][0]
        x_max_curr = bbox[1][0]
        y_min_curr = bbox[0][1]
        y_max_curr = bbox[2][1]

        if not current_phrase_text:
            # Start a new phrase
            current_phrase_text = text
            current_phrase_bbox = bbox
        else:
            # Extract coordinates for previous text in current phrase
            x_min_prev = current_phrase_bbox[0][0]
            x_max_prev = current_phrase_bbox[1][0]
            y_min_prev = current_phrase_bbox[0][1]
            y_max_prev = current_phrase_bbox[2][1]

            # Check if current text is horizontally close to the previous text in the phrase
            # and roughly on the same line (y-coordinates overlap)
            if (x_min_curr - x_max_prev < x_threshold) and \
               (max(y_min_curr, y_min_prev) < min(y_max_curr, y_max_prev) + (y_max_prev - y_min_prev) / 2):
                # Combine with previous text
                current_phrase_text += " " + text
                # Update bounding box to encompass both
                current_phrase_bbox = [
                    [min(x_min_prev, x_min_curr), min(y_min_prev, y_min_curr)],
                    [max(x_max_prev, x_max_curr), min(y_min_prev, y_min_curr)],
                    [max(x_max_prev, x_max_curr), max(y_max_prev, y_max_curr)],
                    [min(x_min_prev, x_min_curr), max(y_max_prev, y_max_curr)]
                ]
            else:
                # Current text is not consecutive, so save the current phrase and start a new one
                combined_phrases.append((current_phrase_bbox, current_phrase_text, 1.0)) # Use 1.0 for combined prob
                current_phrase_text = text
                current_phrase_bbox = bbox
    
    # Add the last phrase
    if current_phrase_text:
        combined_phrases.append((current_phrase_bbox, current_phrase_text, 1.0))

    return combined_phrases

def generate_card_names_config():
    """
    Generates a card_names.csv file by attempting to OCR card names from processed images.
    This overwrites any existing card_names.csv.
    """
    card_names_file = os.path.join('config', 'card_names.csv')
    processed_cards_dir = os.path.join('data', 'processed_cards')
    cards_by_name_path = os.path.join('config', 'cards_by_name.csv')

    # If card_names.csv exists, we will read it and append new entries, not overwrite.
    # If it doesn't exist, a new one will be created.
    existing_card_entries = {}
    if os.path.exists(card_names_file):
        with open(card_names_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for row in reader:
                if row:
                    existing_card_entries[row[0]] = {'file_name': row[0], 'name': row[1], 'type_1': row[2], 'type_2': row[3], 'suit': row[4]}
        print(f"Loaded {len(existing_card_entries)} existing entries from {card_names_file}.")

    # Initialize EasyOCR reader (commented out)
    # reader = easyocr.Reader(['en'], gpu=True, recognizer='Transformer', quantize=True)

    # Load known card names from card_data_manager
    card_collection = CardCollection(cards_by_name_path)
    # Reverted: normalized_known_card_names = {_normalize_text(card.card_name): card.card_name.lower() for card in card_collection.get_all_cards()}
    # Load known card names from card_data_manager
    card_collection = CardCollection(cards_by_name_path)
    # known_card_names_list will contain the original (unnormalized) card names
    known_card_names_list = [card.card_name for card in card_collection.get_all_cards()]

    # Load card layout configurations from card_data.csv
    card_layout_configs = {}
    card_data_file = os.path.join('config', 'card_data.csv')
    if os.path.exists(card_data_file):
        with open(card_data_file, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for row in csv_reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) >= 4: # Ensure enough columns
                    card_type, row_str, col_str, layout = row[:4]
                    try:
                        row_idx = int(row_str)
                        col_idx = int(col_str)
                        card_layout_configs[(card_type, row_idx, col_idx)] = layout.lower()
                    except ValueError:
                        print(f"Skipping line with invalid row/col in card_data.csv: {row}")
    else:
        print(f"Warning: {card_data_file} not found. Cannot determine card layouts.")

    all_card_entries = []

    # Walk through the processed_cards directory to find all image files
    for root, _, files in os.walk(processed_cards_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_name = os.path.basename(file)
                image_path = os.path.join(root, file)
                
                card_name_ocr = ""
                
                # Determine if OCR should be run on this file
                should_run_ocr = False
                
                # Check if it's a split card (ends with _left, _right, _top, _bottom)
                is_split_card = file_name.endswith(('_left.jpg', '_right.jpg', '_top.jpg', '_bottom.jpg'))
                
                if is_split_card:
                    should_run_ocr = True
                else:
                    # It's a base card (e.g., trees_0_0.jpg)
                    # Extract card_type, row, col from file_name (e.g., trees_0_0.jpg -> trees, 0, 0)
                    base_name_without_ext = os.path.splitext(file_name)[0]
                    parts = base_name_without_ext.split('_')
                    if len(parts) == 3: # Expecting format like card_type_row_col
                        card_type = parts[0]
                        try:
                            row_idx = int(parts[1])
                            col_idx = int(parts[2])
                            layout = card_layout_configs.get((card_type, row_idx, col_idx))
                            if layout == "whole":
                                should_run_ocr = True
                        except ValueError:
                            pass # Malformed filename, skip
                
                # if should_run_ocr: # OCR is temporarily disabled
                #     try:
                #         # Load image using PIL and convert to numpy array for EasyOCR
                #         img = Image.open(image_path).convert('RGB')
                #         img_np = np.array(img)

                #         # Image Preprocessing for OCR
                #         gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                #         # Apply Otsu's binarization
                #         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                #         # Perform OCR on the preprocessed image
                #         raw_ocr_results = reader.readtext(binary, detail=1, mag_ratio=1.5) # Use detail=1 for bbox
                #         combined_ocr_results = _combine_ocr_results(raw_ocr_results)

                #         best_match_name_original = ""
                #         best_match_score = 0

                #         for (bbox, text, prob) in combined_ocr_results: # Iterate through combined results
                #             cleaned_text = _normalize_text(text)
                #             print(f"DEBUG: OCR cleaned_text: '{cleaned_text}' for file: {file_name}")
                            
                #             # Generate all possible phrases from cleaned_text
                #             phrases_to_check = []
                #             words = cleaned_text.split()
                #             for i in range(len(words)):
                #                 for j in range(i + 1, min(i + 4, len(words) + 1)): # Phrases up to 3 words long
                #                     phrases_to_check.append(" ".join(words[i:j]))
                            
                #             # Also add the full cleaned_text as a phrase if not already included
                #             if cleaned_text not in phrases_to_check:
                #                 phrases_to_check.append(cleaned_text)

                #             for phrase_to_match in phrases_to_check:
                #                 if not phrase_to_match:
                #                     continue

                #                 # Prioritize exact matches
                #                 # Check if the normalized phrase_to_match exists in our normalized list
                #                 for original_known_name in known_card_names_list:
                #                     if phrase_to_match == _normalize_text(original_known_name):
                #                         if 100 > best_match_score: # Exact match is always 100
                #                             best_match_score = 100
                #                             best_match_name_original = original_known_name
                #                             break # Found an exact match, no need to check further for this OCR result
                #                 # If an exact match was found, break from this inner loop too
                #                 if best_match_name_original and best_match_score == 100:
                #                     break

                #                 # Use rapidfuzz for fuzzy matching
                #                 match = process.extractOne(phrase_to_match, known_card_names_list, scorer=fuzz.WRatio, score_cutoff=70, processor=_normalize_text)
                                
                #                 if match:
                #                     matched_name_original, score, _ = match
                #                     if file_name == 'hCards_0_6_right.jpg':
                #                         print(f"DEBUG: hCards_0_6_right.jpg - Phrase: '{phrase_to_match}', Matched: '{matched_name_original}', Score: {score}")
                #                     # Prioritize higher score, then longer matched name
                #                     if score > best_match_score:
                #                         best_match_score = score
                #                         best_match_name_original = matched_name_original
                #                     elif score == best_match_score:
                #                         # If scores are equal, prioritize longer phrase_to_match
                #                         if len(phrase_to_match) > len(_normalize_text(best_match_name_original)):
                #                             best_match_name_original = matched_name_original
                            
                #             if best_match_name_original and best_match_score == 100: # If exact match found, break from outer loop
                #                 break
                        
                #         card_name_ocr = best_match_name_original if best_match_name_original else ""

                #         print(f"DEBUG: Final card_name_ocr for {file_name}: '{card_name_ocr}'")

                #     except Exception as e:
                #         print(f"Error processing OCR for {file_name}: {e}")
                
                if file_name not in existing_card_entries:
                    all_card_entries.append({'file_name': file_name, 'name': card_name_ocr, 'type_1': '', 'type_2': '', 'suit': ''})

    # Combine existing entries with newly found ones
    final_card_entries = list(existing_card_entries.values()) + all_card_entries

    print(f"DEBUG: All card entries before writing to CSV: {final_card_entries}") # Added print for all_card_entries

    # Write the new data to card_names.csv
    with open(card_names_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'name', 'type_1', 'type_2', 'suit'])
        
        # Sort entries by filename for consistent output
        for entry in sorted(final_card_entries, key=lambda x: x['file_name']):
            writer.writerow([entry['file_name'], entry['name'], entry['type_1'], entry['type_2'], entry['suit']])

    print(f"Finished generating card_names.csv with OCR-extracted names using direct matching.")

if __name__ == '__main__':
    generate_card_names_config()