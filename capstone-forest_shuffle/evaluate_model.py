import os
import json
import torch
import argparse
from sklearn.model_selection import train_test_split
from card_predictor import CardPredictor # Import CardPredictor

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained multi-output model on the validation set.')
    parser.add_argument('--verbose_eval', action='store_true', help='Print individual predictions during evaluation.')
    args = parser.parse_args()

    predictor = CardPredictor() # Initialize the predictor

    # The evaluation logic from CardPredictor.evaluate()
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

    # Convert labels in manifest_data to indices for comparison
    for item in manifest_data:
        for key in predictor.active_label_categories:
            if key == 'types':
                type_indices = [predictor.label_maps['types'].index(t) for t in item['labels']['types'] if t != 'None']
                multi_hot_types = torch.zeros(len(predictor.label_maps['types']), dtype=torch.float)
                multi_hot_types[type_indices] = 1.0
                item['labels'][key] = multi_hot_types
            else:
                label = item['labels'][key]
                if label != 'None':
                    item['labels'][key] = predictor.label_maps[key].index(label)
                else:
                    item['labels'][key] = -1

    _, val_data = train_test_split(manifest_data, test_size=test_size, random_state=random_state)

    correct_predictions = {key: 0 for key in predictor.active_label_categories}
    total_samples = len(val_data)

    for item in val_data:
        image_path = item['file_path']
        ground_truth_labels = item['labels']
        
        predictions = predictor.predict(image_path) # Use the class's predict method

        if verbose_eval:
            # Create a human-readable version of ground_truth_labels for printing
            readable_ground_truth = {}
            for key in predictor.active_label_categories:
                value = ground_truth_labels[key]
                if key == 'name' and value != -1:
                    readable_ground_truth[key] = predictor.label_maps[key][value]
                elif key == 'types':
                    # For types, convert multi-hot tensor back to list of names
                    readable_types = [predictor.label_maps[key][i] for i, v in enumerate(value) if v == 1.0]
                    readable_ground_truth[key] = readable_types if readable_types else ["None"]
                elif value != -1:
                    readable_ground_truth[key] = predictor.label_maps[key][value]
                else:
                    readable_ground_truth[key] = "None"

            print(f"\nImage: {image_path}")
            print(f"  Ground Truth: {readable_ground_truth}")
            print(f"  Prediction:   {predictions}")

        for key in predictor.active_label_categories:
            if key == 'types':
                predicted_multi_hot = torch.zeros(len(predictor.label_maps['types']), dtype=torch.float)
                for p_label in predictions[key]:
                    if p_label != "None":
                        predicted_multi_hot[predictor.label_maps['types'].index(p_label)] = 1.0
                
                if torch.equal(predicted_multi_hot, ground_truth_labels[key]):
                    correct_predictions[key] += 1
            else:
                # Only compare if ground_truth_labels[key] is not -1 (i.e., a valid label)
                if ground_truth_labels[key] != -1 and predictions[key] == predictor.label_maps[key][ground_truth_labels[key]]:
                    correct_predictions[key] += 1
    
    print("\n--- Evaluation Results ---")
    for key in predictor.active_label_categories:
        accuracy = correct_predictions[key] / total_samples
        print(f"{key.capitalize()} Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()