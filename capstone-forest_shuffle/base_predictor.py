import torch
from abc import ABC, abstractmethod
from multi_card_image_processor import MultiCardImageProcessor
from PIL import Image
import numpy as np

class BasePredictor(ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"
        print(f"Using device: {self.device}")

    def is_blank(self, pil_image: Image.Image, threshold=5.0) -> bool:
        """Checks if an image is mostly a single color."""
        if pil_image is None:
            return True
        # Convert to grayscale and to a numpy array
        image_np = np.array(pil_image.convert('L'))
        # If the standard deviation is very low, the image is likely blank
        return image_np.std() < threshold

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict_single_card(self, pil_image: Image.Image):
        pass

    def predict_multi_card_image(self, multi_card_image_path: str, layout_name=None):
        """
        Processes an image potentially containing multiple cards and returns predictions for each.
        """
        processor = MultiCardImageProcessor() # Initialize the processor
        cropped_cards = processor.process_image(multi_card_image_path, layout_name=layout_name)

        all_predictions = []
        for i, card_pil_image in enumerate(cropped_cards):
            print(f"Predicting for cropped card {i+1}...")
            # Save the cropped card to disk for debugging
            card_pil_image.save(f"debug_cropped_card_{i+1}.jpg")  
            
            predictions = self.predict_single_card(card_pil_image)
            all_predictions.append(predictions)
        return all_predictions
