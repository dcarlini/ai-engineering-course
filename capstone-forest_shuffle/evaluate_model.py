import os
import json
import torch
import argparse
from sklearn.model_selection import train_test_split
from model_predictor import ModelPredictor
from clip_predictor import ClipPredictor
from base_predictor import BasePredictor # New import
from PIL import Image # Import Image for predict_single_card
from layout_configs import PREDEFINED_LAYOUTS # New import

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained multi-output model or CLIP on the validation set.')
    parser.add_argument('--verbose_eval', action='store_true', help='Print individual predictions during evaluation.')
    parser.add_argument('--use_clip', action='store_true', help='Use CLIP for prediction instead of the trained multi-output model.')
    parser.add_argument('--multi_card_image_path', type=str, help='Path to an image file containing multiple cards for testing.')
    parser.add_argument('--layout_name', type=str, help='Name of a predefined layout from layout_configs.py (e.g., "horizontal_split").')
    args = parser.parse_args()

    if args.use_clip:
        predictor: BasePredictor = ClipPredictor()
        active_label_categories = ['name', 'types', 'suit', 'expansion']
    else:
        predictor: BasePredictor = ModelPredictor()
        # ModelPredictor has these attributes after _load_model
        active_label_categories = predictor.active_label_categories

    if args.multi_card_image_path:
        if not args.layout_name:
            print("Error: --layout_name is required for --multi_card_image_path.")
            return
        if args.layout_name not in PREDEFINED_LAYOUTS:
            print(f"Error: Unknown layout name: {args.layout_name}. Available layouts: {list(PREDEFINED_LAYOUTS.keys())}")
            return
        
        all_predictions = predictor.predict_multi_card_image(args.multi_card_image_path, layout_config=PREDEFINED_LAYOUTS[args.layout_name])
        print("\n--- Multi-Card Image Prediction Results ---")
        for i, predictions in enumerate(all_predictions):
            print(f"Card {i+1}:")
            for category, prediction in predictions.items():
                if isinstance(prediction, list):
                    print(f"  {category.capitalize()}: {', '.join(prediction) if prediction else 'None'}")
                else:
                    print(f"  {category.capitalize()}: {prediction}")
        return # Exit after multi-card prediction

    manifest_path = 'data/manifest.json'
    test_size = 0.2
    random_state = 42
    verbose_eval = args.verbose_eval

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}. Required for evaluation.")

    print("\n--- Running Evaluation ---")
    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)
    
    # Filter out items where 'name' is 'None' as they are not valid training samples
    manifest_data = [item for item in manifest_data if item['labels']['name'] != 'None']

    # Prepare ground truth labels for comparison
    # For ModelPredictor, convert to indices. For ClipPredictor, keep as strings/lists.
    processed_manifest_data = []
    for item in manifest_data:
        processed_item = item.copy()
        if not args.use_clip:
            for key in active_label_categories:
                if key == 'types':
                    type_indices = [predictor.label_maps['types'].index(t) for t in item['labels']['types'] if t != 'None']
                    multi_hot_types = torch.zeros(len(predictor.label_maps['types']), dtype=torch.float)
                    multi_hot_types[type_indices] = 1.0
                    processed_item['labels'][key] = multi_hot_types
                else:
                    label = item['labels'][key]
                    if label != 'None':
                        processed_item['labels'][key] = predictor.label_maps[key].index(label)
                    else:
                        processed_item['labels'][key] = -1
        processed_manifest_data.append(processed_item)

    _, val_data = train_test_split(processed_manifest_data, test_size=test_size, random_state=random_state)

    correct_predictions = {key: 0 for key in active_label_categories}
    total_samples = len(val_data)

    for item in val_data:
        image_path = item['file_path']
        ground_truth_labels = item['labels']
        
        # Use predict_single_card from the BasePredictor interface
        predictions = predictor.predict_single_card(Image.open(image_path))

        if verbose_eval:
            readable_ground_truth = {}
            for key in active_label_categories:
                value = ground_truth_labels[key]
                if args.use_clip: # CLIP predictions are already strings/lists
                    readable_ground_truth[key] = value
                elif key == 'name' and value != -1:
                    readable_ground_truth[key] = predictor.label_maps[key][value]
                elif key == 'types':
                    readable_types = [predictor.label_maps[key][i] for i, v in enumerate(value) if v == 1.0]
                    readable_ground_truth[key] = readable_types if readable_types else ["None"]
                elif value != -1:
                    readable_ground_truth[key] = predictor.label_maps[key][value]
                else:
                    readable_ground_truth[key] = "None"

            print(f"\nImage: {image_path}")
            print(f"  Ground Truth: {readable_ground_truth}")
            print(f"  Prediction:   {predictions}")

        for key in active_label_categories:
            if args.use_clip:
                # For CLIP, predictions[key] is already the string/list label
                # Handle 'types' as a list comparison
                if key == 'types':
                    # Ensure both are sorted for consistent comparison
                    gt_types = sorted([t for t in ground_truth_labels.get(key, []) if t != 'None'])
                    pred_types = sorted([t for t in predictions.get(key, []) if t != 'None'])
                    if gt_types == pred_types:
                        correct_predictions[key] += 1
                else:
                    if predictions.get(key) == ground_truth_labels.get(key):
                        correct_predictions[key] += 1
            else: # For MultiOutputModel
                if key == 'types':
                    predicted_multi_hot = torch.zeros(len(predictor.label_maps['types']), dtype=torch.float)
                    for p_label in predictions[key]: # predictions[key] is a list of strings here
                        if p_label != "None":
                            predicted_multi_hot[predictor.label_maps['types'].index(p_label)] = 1.0
                    
                    if torch.equal(predicted_multi_hot, ground_truth_labels[key]):
                        correct_predictions[key] += 1
                else:
                    # Only compare if ground_truth_labels[key] is not -1 (i.e., a valid label)
                    # predictions[key] is a string, ground_truth_labels[key] is an index
                    if ground_truth_labels[key] != -1 and predictions[key] == predictor.label_maps[key][ground_truth_labels[key]]:
                        correct_predictions[key] += 1
    
    print("\n--- Evaluation Results ---")
    for key in active_label_categories:
        accuracy = correct_predictions[key] / total_samples
        print(f"{key.capitalize()} Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()