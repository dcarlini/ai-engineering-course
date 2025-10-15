import torch
import clip
from PIL import Image
import os
import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from base_predictor import BasePredictor # New import

class ClipPredictor(BasePredictor):
    def __init__(self, manifest_path='data/manifest.json'):
        super().__init__() # Initialize BasePredictor
        self.manifest_path = manifest_path
        self._load_model() # Call _load_model after manifest_path is set

    def _load_model(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.precomputed_image_features, self.precomputed_labels = self._precompute_image_features_and_labels()

    def _precompute_image_features_and_labels(self):
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Please run data_preparation.py first.")
        
        with open(self.manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        image_features_list = []
        labels_list = []

        print("Precomputing CLIP image features from manifest...")
        for item in manifest_data:
            image_path = item['file_path']
            card_name = item['labels']['name']

            if card_name == 'None': # Skip items without a valid name
                continue

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}. Skipping.")
                continue

            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image)
            image_features_list.append(image_feature)
            
            # Store all relevant labels as a dictionary
            labels_list.append({
                'name': item['labels'].get('name', 'None'),
                'types': item['labels'].get('types', []), # types is a list
                'suit': item['labels'].get('suit', 'None'),
                'expansion': item['labels'].get('expansion', 'None')
            })
        
        if not image_features_list:
            raise ValueError("No valid images found in manifest to precompute features.")

        precomputed_image_features = torch.cat(image_features_list)
        precomputed_image_features /= precomputed_image_features.norm(dim=-1, keepdim=True)
        print(f"Precomputed {len(labels_list)} image features.")
        return precomputed_image_features, labels_list

    def predict_single_card(self, pil_image: Image.Image):
        """Predicts labels for a single PIL Image object."""
        if self.is_blank(pil_image):
            return {}

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_image_feature = self.model.encode_image(image)
        query_image_feature /= query_image_feature.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between query image feature and precomputed image features
        similarity = (100.0 * query_image_feature @ self.precomputed_image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # Return the full dictionary of labels for the closest image
        predicted_labels = self.precomputed_labels[indices[0].item()]
        return predicted_labels

