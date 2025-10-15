import argparse
import json
from PIL import Image

from model_predictor import ModelPredictor
from clip_predictor import ClipPredictor
from base_predictor import BasePredictor
from layout_configs import PREDEFINED_LAYOUTS # New import

def main():
    parser = argparse.ArgumentParser(description='Predict card attributes from an image using either a trained multi-output model or CLIP.')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument('--layout_name', type=str, help='Name of a predefined layout from layout_configs.py (e.g., "horizontal_split"). If provided, the image is treated as a multi-card image.')
    parser.add_argument('--use_clip', action='store_true', help='Use CLIP for prediction instead of the trained multi-output model.')
    args = parser.parse_args()

    if args.use_clip:
        predictor: BasePredictor = ClipPredictor()
    else:
        predictor: BasePredictor = ModelPredictor()

    if args.layout_name:
        if args.layout_name not in PREDEFINED_LAYOUTS:
            print(f"Error: Unknown layout name: {args.layout_name}. Available layouts: {list(PREDEFINED_LAYOUTS.keys())}")
            return
        
        all_predictions = predictor.predict_multi_card_image(args.image_path, layout_name=args.layout_name)
        if not all_predictions:
            print("No cards detected or processed in the multi-card image with the given layout.")
            return
        print("\n--- Multi-Card Image Prediction Results ---")
        for i, predictions in enumerate(all_predictions):
            print(f"Card {i+1}:")
            for category, prediction in predictions.items():
                if isinstance(prediction, list):
                    print(f"  {category.capitalize()}: {', '.join(prediction) if prediction else 'None'}")
                else:
                    print(f"  {category.capitalize()}: {prediction}")
    else:
        predictions = predictor.predict_single_card(Image.open(args.image_path))
        print("\n--- Prediction Results ---")
        for category, prediction in predictions.items():
            if isinstance(prediction, list):
                print(f"{category.capitalize()}: {', '.join(prediction) if prediction else 'None'}")
            else:
                print(f"{category.capitalize()}: {prediction}")

if __name__ == '__main__':
    main()
