# Forest Shuffle Card Recognition

This project aims to develop a machine learning model to recognize and classify cards from the "Forest Shuffle" board game. It includes a pipeline for data preparation, multi-output model training, single-image prediction, and model evaluation.

## Project Structure

-   `card_data_manager.py`: Manages card data loaded from CSV files.
-   `symbol_manager.py`: Manages symbol data loaded from CSV files.
-   `data_preparation.py`: Downloads card images, processes them into individual card images, and generates a manifest file (`data/manifest.json`) for training.
-   `model_architecture.py`: Defines the `MultiOutputModel` architecture, which uses a ResNet backbone with multiple output heads for different card attributes.
-   `train_multi_output_model.py`: Script for training the multi-output model.
-   `model_predictor.py`: Script for making predictions on single card images using the trained multi-output model.
-   `evaluate_model.py`: Script for evaluating the trained model's performance on a validation set.
-   `clip_predictor.py`: Script for making predictions on single card images using the CLIP model (image-to-image retrieval).
-   `requirements.txt`: Lists all Python dependencies.
-   `config/`: Directory containing configuration CSVs and the training configuration JSON.
    -   `card_data.csv`: Defines how card sheets are split into individual cards.
    -   `card_names.csv`: Maps image filenames to card names and other attributes.
    -   `card_sets.csv`: Contains URLs for raw card image sheets.
    -   `cards_by_name.csv`: Master list of all cards and their properties.
    -   `symbols.csv`: Defines game symbols.
    -   `training_config.json`: Configuration for model training parameters.
-   `data/`: Directory for generated data (raw images, processed cards, manifest).
    -   `manifest.json`: A JSON file mapping image paths to their corresponding labels.
-   `models/`: Directory for saving trained models and label mappings.
    -   `multi_output_model.pth`: Trained model weights.
    -   `label_maps.json`: Mappings from label indices to human-readable labels.

## CLIP Integration (Image-to-Image Retrieval)

This project also integrates [CLIP (Contrastive Language-Image Pre-training)](https://openai.com/research/clip) for card recognition using an image-to-image retrieval approach. This method leverages CLIP's powerful ability to understand visual similarity by comparing an input image directly to a database of pre-encoded card images. This approach has demonstrated 100% accuracy on the validation set for all card attributes.

## Setup

To set up the project environment, follow these steps:

1.  **Create a virtual environment** (if you don't have one):
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies** (including `openai-clip`):
    ```bash
    .venv/bin/pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

Run the data preparation script to download raw card images, process them, and generate the `manifest.json` file. This script will clean up previous `data/` directories before running.

```bash
.venv/bin/python data_preparation.py
```

### 2. Model Training (Multi-Output Model)

Train the multi-output model using the prepared data. Training parameters like epochs, learning rate, and active label categories are configured in `config/training_config.json`.

Edit `config/training_config.json` to adjust training parameters. For example, to focus on 'name' recognition:

```json
{
  "epochs": 25,
  "learning_rate": 0.0001,
  "batch_size": 32,
  "optimizer": "Adam",
  "label_categories": ["name"]
}
```

Then, run the training script:

```bash
.venv/bin/python train_multi_output_model.py
```

The trained model (`multi_output_model.pth`) and label mappings (`label_maps.json`) will be saved in the `models/` directory.

### 3. Model Evaluation

Evaluate the trained model or the CLIP model on the validation set. You can choose to see a summary or verbose output with individual predictions.

**Evaluate Multi-Output Model:**

To run a summary evaluation:

```bash
.venv/bin/python evaluate_model.py
```

To see individual predictions and ground truth during evaluation:

```bash
.venv/bin/python evaluate_model.py --verbose_eval
```

**Evaluate CLIP Model:**

To run a summary evaluation using CLIP:

```bash
.venv/bin/python evaluate_model.py --use_clip
```

To see individual predictions and ground truth during evaluation using CLIP:

```bash
.venv/bin/python evaluate_model.py --use_clip --verbose_eval
```

### 4. Single Image Prediction

Use either the trained multi-output model or the CLIP model to predict attributes for a single image.

**Predict with Trained Multi-Output Model:**

```bash
.venv/bin/python model_predictor.py <path_to_your_image.jpg>
```

**Predict with RAG using CLIP Model:**

```bash
.venv/bin/python clip_predictor.py <path_to_your_image.jpg>
```

Replace `<path_to_your_image.jpg>` with the actual path to the image file you want to predict on.

## Future Improvements

-   **Improve `types` accuracy:** Investigate the low accuracy for multi-label 'types' prediction. This might involve adjusting the loss function, model architecture, or data augmentation strategies.
-   **Re-introduce `suit` prediction:** If `suit` information becomes available or is deemed important, re-integrate it into the training and prediction pipeline.
-   **Web Service Integration:** Develop a simple web service (e.g., using Flask or FastAPI) to expose the prediction functionality via an API.
-   **More Robust Evaluation Metrics:** Implement more advanced multi-label evaluation metrics (e.g., F1-score, Jaccard index) for the 'types' category.
-   **Hyperparameter Tuning:** Optimize training parameters (learning rate, batch size, epochs) using techniques like grid search or random search.