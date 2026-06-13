"""
Data Preparation and Dataset Management
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432

This module helps with preparing datasets for YOLOv8 training and inference.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manage datasets for YOLOv8 training and inference"""
    
    def __init__(self, dataset_path: str = 'data/'):
        """
        Initialize DatasetManager
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetManager initialized at {dataset_path}")
    
    def create_dataset_structure(self, dataset_name: str) -> Path:
        """
        Create standard YOLOv8 dataset structure
        
        Expected structure:
        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Path to created dataset
        """
        dataset_dir = self.dataset_path / dataset_name
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset structure created at {dataset_dir}")
        return dataset_dir
    
    def create_data_yaml(self, 
                        dataset_path: str,
                        num_classes: int,
                        class_names: List[str],
                        train_path: str = 'images/train',
                        val_path: str = 'images/val',
                        test_path: Optional[str] = None) -> str:
        """
        Create data.yaml configuration file for YOLOv8
        
        Args:
            dataset_path: Path to dataset
            num_classes: Number of classes
            class_names: List of class names
            train_path: Path to training images
            val_path: Path to validation images
            test_path: Path to test images (optional)
            
        Returns:
            Path to created data.yaml
        """
        data_yaml = {
            'path': str(Path(dataset_path).absolute()),
            'train': train_path,
            'val': val_path,
            'test': test_path if test_path else None,
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        # Remove test if not provided
        if data_yaml['test'] is None:
            del data_yaml['test']
        
        # Save to YAML
        yaml_path = Path(dataset_path) / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"data.yaml created at {yaml_path}")
        return str(yaml_path)
    
    def organize_images(self, 
                       source_dir: str,
                       dataset_dir: str,
                       split: str = 'train',
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.2) -> Dict:
        """
        Organize images from source directory into dataset structure
        
        Args:
            source_dir: Source directory with images
            dataset_dir: Dataset directory
            split: Which split to use ('train', 'val', 'test')
            train_ratio: Ratio for training (if splitting)
            val_ratio: Ratio for validation (if splitting)
            
        Returns:
            Dictionary with operation statistics
        """
        source_path = Path(source_dir)
        dataset_path = Path(dataset_dir)
        
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return {'success': False, 'error': 'Source directory not found'}
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        stats = {
            'total_files': len(image_files),
            'copied': 0,
            'failed': 0,
            'splits': {}
        }
        
        # Copy images
        target_dir = dataset_path / 'images' / split
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for image_file in image_files:
            try:
                shutil.copy2(image_file, target_dir / image_file.name)
                stats['copied'] += 1
            except Exception as e:
                logger.warning(f"Failed to copy {image_file.name}: {str(e)}")
                stats['failed'] += 1
        
        logger.info(f"Organized {stats['copied']} images into {split} split")
        return stats
    
    def validate_dataset(self, dataset_path: str) -> Dict:
        """
        Validate dataset structure and contents
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Validation report
        """
        dataset_path = Path(dataset_path)
        report = {
            'valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check data.yaml
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            report['issues'].append("data.yaml not found")
            report['valid'] = False
        
        # Check directory structure
        for split in ['train', 'val']:
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if not images_dir.exists():
                report['issues'].append(f"Missing images/{split} directory")
                report['valid'] = False
            else:
                image_count = len(list(images_dir.glob('*')))
                report['statistics'][f'{split}_images'] = image_count
            
            if not labels_dir.exists():
                report['issues'].append(f"Missing labels/{split} directory")
                report['valid'] = False
            else:
                label_count = len(list(labels_dir.glob('*.txt')))
                report['statistics'][f'{split}_labels'] = label_count
        
        return report
    
    def split_dataset(self, 
                     source_dir: str,
                     dataset_dir: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.2) -> Dict:
        """
        Split dataset into train/val/test sets
        
        Args:
            source_dir: Source directory with all images
            dataset_dir: Target dataset directory
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Statistics about split operation
        """
        import random
        
        source_path = Path(source_dir)
        dataset_path = Path(dataset_dir)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_idx = int(total * train_ratio)
        val_idx = train_idx + int(total * val_ratio)
        
        # Split files
        train_files = image_files[:train_idx]
        val_files = image_files[train_idx:val_idx]
        test_files = image_files[val_idx:]
        
        stats = {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        
        # Copy files to respective directories
        for split_name, split_files in [('train', train_files), 
                                        ('val', val_files), 
                                        ('test', test_files)]:
            if split_files:
                self.organize_images(source_dir, dataset_dir, split_name)
        
        logger.info(f"Dataset split: Train={len(train_files)}, "
                   f"Val={len(val_files)}, Test={len(test_files)}")
        
        return stats


def create_sample_dataset_config():
    """Create a sample data.yaml configuration"""
    sample_config = {
        'path': '/path/to/dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 80,  # Number of classes
        'names': {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            # ... Add more class names
        }
    }
    
    config_path = 'data/sample_data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    
    logger.info(f"Sample config created at {config_path}")
    return config_path


def main():
    """Example usage"""
    # Initialize manager
    manager = DatasetManager('data/datasets')
    
    # Create dataset structure
    dataset_dir = manager.create_dataset_structure('my_dataset')
    
    # Create data.yaml
    class_names = ['person', 'car', 'bicycle']
    manager.create_data_yaml(
        str(dataset_dir),
        num_classes=3,
        class_names=class_names
    )
    
    # Validate dataset
    report = manager.validate_dataset(str(dataset_dir))
    print(f"Validation report: {report}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
