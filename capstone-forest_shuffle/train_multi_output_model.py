
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import argparse
import time
from sklearn.model_selection import train_test_split
from model_architecture import MultiOutputModel # Added import

class MultiLabelDataset(Dataset):
    """Custom dataset for multi-label classification."""
    def __init__(self, manifest_data, label_categories, label_maps, transform=None):
        self.manifest_data = manifest_data
        self.label_categories = label_categories
        self.label_maps = label_maps # Store label_maps
        self.transform = transform

    def __len__(self):
        return len(self.manifest_data)

    def __getitem__(self, idx):
        item = self.manifest_data[idx]
        image = Image.open(item['file_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = {}
        for key in self.label_categories:
            if key == 'types':
                # Create a multi-hot encoded tensor for 'types'
                type_indices = [self.label_maps['types'].index(t) for t in item['labels']['types'] if t != 'None']
                multi_hot_types = torch.zeros(len(self.label_maps['types']), dtype=torch.float)
                multi_hot_types[type_indices] = 1.0
                labels[key] = multi_hot_types
            else:
                labels[key] = item['labels'][key]
        
        return image, labels

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25):
    """Trains the multi-output model."""
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = {key: 0 for key in model.heads.keys()}
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                
                # Convert label names to integer indices
                label_indices = {}
                for key, val in labels.items():
                    if key == 'types':
                        label_indices[key] = val.to(device) # types is already multi-hot
                    else:
                        label_indices[key] = val.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    losses = {}
                    for key in model.heads.keys():
                        if key == 'types':
                            losses[key] = criterion['types'](outputs[key], label_indices[key])
                        else:
                            losses[key] = criterion['other'](outputs[key], label_indices[key])
                    total_loss = sum(losses.values())

                    if phase == 'train':
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Add gradient clipping
                        optimizer.step()

                running_loss += total_loss.item() * inputs.size(0)
                for key in model.heads.keys():
                    if key == 'types':
                        # For multi-label, accuracy is more complex.
                        # Let's use a threshold of 0.5 and count exact matches for simplicity.
                        preds = (torch.sigmoid(outputs[key]) > 0.5).float()
                        running_corrects[key] += torch.sum(torch.all(preds == label_indices[key], dim=1)).item()
                    else:
                        _, preds = torch.max(outputs[key], 1)
                        running_corrects[key] += torch.sum(preds == label_indices[key].data)
                
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_accs = {key: float(running_corrects[key]) / total_samples if key == 'types' else running_corrects[key].float() / total_samples for key in model.heads.keys()}

            print(f'{phase} Loss: {epoch_loss:.4f}')
            for key, acc in epoch_accs.items():
                print(f'  {key} Acc: {acc:.4f}')

            # Use average accuracy for selecting the best model
            avg_acc = sum(epoch_accs.values()) / len(epoch_accs)
            if phase == 'val' and avg_acc > best_acc:
                best_acc = avg_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Avg Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    # Load training configuration
    config_path = os.path.join('config', 'training_config.json')
    if not os.path.exists(config_path):
        print(f"Error: Training config file not found at {config_path}.")
        return
    with open(config_path, 'r') as f:
        config = json.load(f)

    epochs = config.get('epochs', 25)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)

    manifest_path = os.path.join('data', 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}. Please run data_preparation.py first.")
        return

    with open(manifest_path, 'r') as f:
        manifest_data = json.load(f)

    # Define all possible label categories
    all_label_categories = config.get('label_categories', ['name', 'types', 'expansion'])
    
    # Create label-to-index mappings and filter out categories with no valid labels
    label_maps = {}
    num_classes_dict = {}
    active_label_categories = []
    for key in all_label_categories:
        if key == 'types':
            # For 'types', collect all unique types from the lists
            all_types = set()
            for item in manifest_data:
                for t in item['labels']['types']:
                    if t != 'None':
                        all_types.add(t)
            unique_labels = sorted(list(all_types))
        else:
            unique_labels = sorted(list(set(item['labels'][key] for item in manifest_data if item['labels'][key] != 'None')))
        
        if unique_labels: # Only include if there are actual labels
            label_maps[key] = unique_labels
            num_classes_dict[key] = len(unique_labels)
            active_label_categories.append(key)
    
    # Convert labels in manifest_data to indices
    for item in manifest_data:
        for key in all_label_categories: # Iterate through all possible categories
            if key == 'types':
                # 'types' are handled as multi-hot encoding in __getitem__
                # No need to convert to index here, just ensure it's a list of strings
                pass
            else:
                label = item['labels'][key]
                if key in label_maps and label != 'None': # Only map if category is active and label is not 'None'
                    item['labels'][key] = label_maps[key].index(label)
                else: # Handle 'None' labels or inactive categories
                    item['labels'][key] = -1 # Will be ignored by loss function if target is -1

    # Filter out items where 'name' is 'None' as they are not valid training samples
    manifest_data = [item for item in manifest_data if item['labels']['name'] != -1]

    # Split data into training and validation sets
    train_data, val_data = train_test_split(manifest_data, test_size=0.2, random_state=42)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': MultiLabelDataset(train_data, active_label_categories, label_maps, transform=data_transforms['train']),
        'val': MultiLabelDataset(val_data, active_label_categories, label_maps, transform=data_transforms['val'])
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = MultiOutputModel(num_classes_dict).to(device)

    criterion = {
        'types': nn.BCEWithLogitsLoss(),
        'other': nn.CrossEntropyLoss(ignore_index=-1) # Ignore -1 labels
    }

    optimizer_name = config.get('optimizer', 'SGD')
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    # Train and save the model
    trained_model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=epochs)
    
    # Ensure models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, 'multi_output_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save label maps
    label_maps_path = os.path.join(models_dir, 'label_maps.json')
    with open(label_maps_path, 'w') as f:
        json.dump(label_maps, f)
    print(f"Label maps saved to {label_maps_path}")

if __name__ == '__main__':
    main()
