import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from skimage import color
import random
import shutil

# --- Transformations ---
class HEDStainJitterTensor:
    """
    H&E stain jitter in HED space.
    Expects: torch Tensor [C,H,W], float in [0,1].
    Returns: same type/device, still in [0,1].
    """
    def __init__(self, p=0.5, sigma_h=0.03, sigma_e=0.03):
        self.p = p
        self.sigma_h = sigma_h
        self.sigma_e = sigma_e

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        device = img.device
        dtype = img.dtype

        x = img.detach().to("cpu").permute(1, 2, 0).numpy()  # HWC
        x = np.clip(x, 0.0, 1.0)

        hed = color.rgb2hed(x)
        hed[..., 0] += np.random.normal(0, self.sigma_h, size=hed[..., 0].shape)  # H
        hed[..., 1] += np.random.normal(0, self.sigma_e, size=hed[..., 1].shape)  # E

        rgb = np.clip(color.hed2rgb(hed), 0.0, 1.0)
        out = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=dtype, device=device)

        return out

class RandomRotate90Tensor:
    def __init__(self, p=0.7):
        self.p = p
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        k = random.randint(0, 3)
        return torch.rot90(img, k, dims=(1, 2))

# --- Dataset Class ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (224, 224)

class TissueDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=None, normalize_imagenet=False, cache_images=False, data_root_override=None):
        self.augmentation = augmentation
        self.normalize_imagenet = normalize_imagenet
        self.df = df
        
        # Determine actual paths
        self.paths = df['path'].tolist()
        if data_root_override:
             # Fix paths if running in different env
             self.paths = [os.path.join(data_root_override, os.path.basename(p)) for p in self.paths]
             
        self.labels = df['label_encoded'].tolist()
        
        self.to_tensor = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        
        if normalize_imagenet:
            self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else:
            self.normalize = None
            
        # Cache disabled for HPO to save memory/startup time in workers, 
        # unless explicitly requested (best to rely on OS page cache for HPO workers)
        self.image_cache = {}
        if cache_images:
             # Simplified caching for HPO if needed
             pass

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            image = self.to_tensor(img)
        except Exception as e:
            # Fallback
            image = torch.zeros((3, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.float32)
        
        if self.augmentation:
            image = self.augmentation(image)
        
        if self.normalize:
            image = self.normalize(image)
        
        return image, label

# --- Model retrieval ---
# Assuming ResNet.py is in the same directory
try:
    from ResNet import resnet50
except ImportError:
    # Try adding current directory
    sys.path.append(os.getcwd())
    try:
        from ResNet import resnet50
    except ImportError:
        print("Error: Could not import ResNet. Ensure ResNet.py is present.")
        sys.exit(1)

def get_model(num_classes=4):
    return resnet50(num_classes=num_classes)

# --- Training Loop (HPO) ---
def train_hpo(config, df_train=None, df_val=None, data_dir=None):
    # Re-import inside worker if needed, though arguments are preferred
    
    # 1. Setup Augmentation (Dynamic)
    aug_rot = config.get('aug_rotation', 0)
    aug_col = config.get('aug_color', 0)
    
    aug_list = [
        HEDStainJitterTensor(p=0.5), # Always keep stain jitter for H&E
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
    
    if aug_rot > 0:
        aug_list.append(transforms.RandomRotation(degrees=aug_rot)) # If tensor, use RandomRotation or custom
    
    # Note: RandomRotate90Tensor is custom, adding it if requested implicitly or always
    aug_list.append(RandomRotate90Tensor(p=0.5))
    
    if aug_col > 0:
        aug_list.append(transforms.ColorJitter(brightness=aug_col, contrast=aug_col, saturation=aug_col, hue=0.02))
        
    train_transform = transforms.Compose(aug_list)
    
    # 2. Datasets
    # We use df passed via parameters
    train_dataset = TissueDataset(df_train, augmentation=train_transform, normalize_imagenet=True, data_root_override=data_dir)
    val_dataset = TissueDataset(df_val, augmentation=None, normalize_imagenet=True, data_root_override=data_dir)
    
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=4)
    model = model.to(device)
    
    # 4. Loss with Class Weights
    # Calculate weights dynamically from passed df_train to be safe
    class_counts = df_train['label_encoded'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    n_classes = len(class_counts)
    
    tuning_factors = torch.tensor([1.0, 1.0, 1.0, 0.6], dtype=torch.float32).to(device)
    base_weights = torch.tensor([total_samples / (n_classes * c) for c in class_counts], dtype=torch.float32).to(device)
    final_weights = base_weights * tuning_factors
    
    criterion = nn.CrossEntropyLoss(weight=final_weights, label_smoothing=config.get('label_smoothing', 0.0))
    
    # 5. Optimizer
    lr = config['lr']
    wd = config['weight_decay']
    # Force AdamW as requested
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # --- Phase 1: Transfer Learning (Head Only) ---
    # Freeze backbone
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Train 5 epochs
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # --- Phase 2: Fine Tuning (Full) ---
    for param in model.parameters():
        param.requires_grad = True
        
    # Re-init optimizer with lower LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=wd)
    
    # Train 10 epochs (or less if pruned)
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_dataset)
        accuracy = correct / total
        
        # Report
        train.report({'loss': val_loss, 'accuracy': accuracy})

# --- Main Execution ---
def create_metadata_dataframe(patches_dir, csv_laws):
    # Minimal reconstruction of metadata creation
    # csv_laws is path to labels.csv
    if not os.path.exists(csv_laws):
        print(f"Error: Labels CSV not found at {csv_laws}")
        return pd.DataFrame()
        
    df_labels = pd.read_csv(csv_laws)
    # Assumes 'filename' or similar and 'label'
    # Adjust based on notebook logic: 'slide_id', 'label'
    # Notebook: id_col = 'sample_id' (from filename), label_col = 'label'
    
    # Let's try to infer patches from directory
    if not os.path.exists(patches_dir):
        print(f"Error: Patches dir not found at {patches_dir}")
        return pd.DataFrame()
        
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.png')]
    data = []
    
    # Rename columns if needed to match code
    if 'filename' in df_labels.columns and 'sample_id' not in df_labels.columns:
         df_labels['sample_id'] = df_labels['filename'].apply(lambda x: os.path.splitext(x)[0])
         
    for filename in patch_files:
        try:
            bag_id = filename.rsplit('_p', 1)[0]
            data.append({
                'filename': filename,
                'sample_id': bag_id,
                'path': os.path.join(patches_dir, filename)
            })
        except:
            continue
            
    df_patches = pd.DataFrame(data)
    df = pd.merge(df_patches, df_labels, on='sample_id', how='inner')
    
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    return df

if __name__ == "__main__":
    # 1. Config paths
    # Try colab paths or local paths
    if os.path.exists('/content/dataset/an2dl2526c2'):
        DATA_ROOT = '/content/dataset/an2dl2526c2'
        CSV_PATH = '/content/dataset/data_split.csv' # Adjust if different
    else:
        # Fallback local or relative
        DATA_ROOT = 'dataset/an2dl2526c2' # Relative to execution
        CSV_PATH = 'dataset/labels.csv' 
        
    # Check if we need to find the dataset
    # We assume notebook extracts to a known location or we scan
    # Let's assume standard AN2DL path structure
    if not os.path.exists(DATA_ROOT):
        # Allow passing as arg?
        pass

    # For now, let's assume the user runs this where data is available or we find it
    # We will look for "patches" folder
    
    # MockING dataset setup for script correctness if paths fail, 
    # but in real run we need valid paths.
    # We will assume the notebook has prepared the data in a folder "patches" or similar.
    
    # Find dataset dir by looking for 'img_*.png'
    # Actually, better to replicate notebook Data Loading logic if possible
    # Notebook uses PATCHES_OUT and CSV_PATH.
    
    # Let's expect the user (notebook) to have defined paths or run in a specific way.
    # We will search for 'patches' folder in current or parent dir
    patches_dir = None
    possible_dirs = ['patches', 'data/patches', '../patches', '/content/dataset/an2dl2526c2/patches']
    for d in possible_dirs:
        if os.path.exists(d):
            patches_dir = d
            break
            
    labels_csv = None
    possible_csvs = ['labels.csv', 'data/labels.csv', '../labels.csv', '/content/dataset/labels.csv']
    for f in possible_csvs:
        if os.path.exists(f):
            labels_csv = f
            break
    
    if not patches_dir or not labels_csv:
        print("Warning: Dataset not found automatically. Please ensure 'patches' dir and 'labels.csv' exist.")
        # We assume arguments might be provided or handled by the notebook wrapper
        # For HPO test, we might exit or fail.
    
    if patches_dir and labels_csv:
        print(f"Loading data from {patches_dir} using {labels_csv}")
        df = create_metadata_dataframe(patches_dir, labels_csv)
        
        # Split
        unique_samples = df['sample_id'].unique()
        train_samples, val_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        df_train = df[df['sample_id'].isin(train_samples)].reset_index(drop=True)
        df_val = df[df['sample_id'].isin(val_samples)].reset_index(drop=True)
        
        # Define Search Space
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "weight_decay": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([32, 64]),
            "aug_rotation": tune.choice([0, 15, 30, 45]),
            "aug_color": tune.choice([0, 0.1, 0.2]),
            "label_smoothing": tune.choice([0.0, 0.1])
        }
        
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=15, # 5 TL + 10 FT
            grace_period=5,
            reduction_factor=2
        )
        
        print("Starting Ray Tune...")
        result = tune.run(
            tune.with_parameters(train_hpo, df_train=df_train, df_val=df_val, data_dir=patches_dir),
            config=config,
            num_samples=10, 
            scheduler=scheduler,
            resources_per_trial={"cpu": 2, "gpu": 1},
            storage_path=os.path.abspath("./ray_results")
        )
        
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
        
        # Save best config
        import json
        with open("best_config.json", "w") as f:
            json.dump(best_trial.config, f)
            
    else:
        print("Cannot run HPO without data.")
