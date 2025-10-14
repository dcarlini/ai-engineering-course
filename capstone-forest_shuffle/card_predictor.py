

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from sklearn.model_selection import train_test_split # Added import
from model_architecture import MultiOutputModel # Added import

class CardPredictor:
    """
    A class to load a trained multi-output model and make predictions.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.label_maps = None
        self.num_classes_dict = None
        self.active_label_categories = None
        self.device = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.models_dir, 'multi_output_model.pth')
        label_maps_path = os.path.join(self.models_dir, 'label_maps.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        if not os.path.exists(label_maps_path):
            raise FileNotFoundError(f"Label maps file not found at {label_maps_path}. Please train the model first.")

        with open(label_maps_path, 'r') as f:
            self.label_maps = json.load(f)

        self.num_classes_dict = {key: len(labels) for key, labels in self.label_maps.items()}
        self.active_label_categories = list(self.label_maps.keys())

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model = MultiOutputModel(self.num_classes_dict).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode

    def predict(self, image_path):
        """Predicts the labels of a single image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)

        predictions = {}
        for key, output_tensor in outputs.items():
            if key == 'types':
                probabilities = torch.sigmoid(output_tensor).squeeze(0)
                predicted_indices = (probabilities > 0.5).nonzero(as_tuple=True)[0].tolist()
                predicted_labels = [self.label_maps[key][i] for i in predicted_indices]
                predictions[key] = predicted_labels if predicted_labels else ["None"]
            else:
                _, predicted_idx = torch.max(output_tensor, 1)
                predictions[key] = self.label_maps[key][predicted_idx.item()]
                
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Predict card attributes from an image using a trained multi-output model.')
    parser.add_argument('image_path', type=str, help='Path to the input image file for single prediction.')
    args = parser.parse_args()

    predictor = CardPredictor() # Initialize the predictor

    predictions = predictor.predict(args.image_path)
    print("\n--- Prediction Results ---")
    for category, prediction in predictions.items():
        print(f"{category.capitalize()}: {prediction}")

if __name__ == '__main__':
    main()
