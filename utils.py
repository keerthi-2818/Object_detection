"""
Utility Functions for Real-Time Object Detection System
Internship ID: CITS2432
Domain: Artificial Intelligence and Computer Vision
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DetectionUtils:
    """Utility class for object detection operations"""
    
    @staticmethod
    def draw_bounding_boxes(image: np.ndarray, 
                           detections: dict, 
                           confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw bounding boxes on image with labels and confidence scores
        
        Args:
            image: Input image
            detections: Dictionary containing detection results
            confidence_threshold: Minimum confidence score
            
        Returns:
            Image with bounding boxes drawn
        """
        annotated_image = image.copy()
        
        if detections is None or len(detections) == 0:
            return annotated_image
        
        # Handle different detection result formats
        if hasattr(detections, 'boxes'):
            boxes = detections.boxes
            for box in boxes:
                if box.conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Put label with confidence
                    label = f"Object {cls_id}: {conf:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return annotated_image
    
    @staticmethod
    def get_detection_stats(detections: dict) -> Dict:
        """
        Extract statistics from detections
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary containing detection statistics
        """
        stats = {
            'total_objects': 0,
            'average_confidence': 0.0,
            'object_classes': {},
            'max_confidence': 0.0,
            'min_confidence': 1.0
        }
        
        if detections is None:
            return stats
        
        if hasattr(detections, 'boxes'):
            boxes = detections.boxes
            confidences = []
            
            for box in boxes:
                stats['total_objects'] += 1
                conf = float(box.conf[0])
                confidences.append(conf)
                
                cls_id = int(box.cls[0])
                if cls_id not in stats['object_classes']:
                    stats['object_classes'][cls_id] = 0
                stats['object_classes'][cls_id] += 1
            
            if confidences:
                stats['average_confidence'] = np.mean(confidences)
                stats['max_confidence'] = np.max(confidences)
                stats['min_confidence'] = np.min(confidences)
        
        return stats
    
    @staticmethod
    def calculate_fps(start_time: float, frame_count: int) -> float:
        """
        Calculate frames per second
        
        Args:
            start_time: Start time
            frame_count: Number of frames processed
            
        Returns:
            FPS value
        """
        import time
        elapsed = time.time() - start_time
        return frame_count / elapsed if elapsed > 0 else 0
    
    @staticmethod
    def create_output_path(output_dir: str, filename: str, extension: str) -> str:
        """
        Create output file path with timestamp
        
        Args:
            output_dir: Output directory
            filename: Base filename
            extension: File extension
            
        Returns:
            Full output path
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(output_dir, f"{filename}_{timestamp}.{extension}")
    
    @staticmethod
    def log_detection(image_path: str, detections: dict, stats: dict):
        """
        Log detection results
        
        Args:
            image_path: Path to image
            detections: Detection results
            stats: Detection statistics
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'stats': stats
        }
        logger.info(f"Detection logged: {json.dumps(log_entry, indent=2)}")
    
    @staticmethod
    def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
        
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class AnalyticsEngine:
    """Engine for analytics and statistics"""
    
    def __init__(self):
        self.detections_history = []
    
    def add_detection(self, stats: Dict):
        """Add detection statistics to history"""
        stats['timestamp'] = datetime.now().isoformat()
        self.detections_history.append(stats)
    
    def get_object_frequency(self) -> Dict:
        """Get frequency of detected objects"""
        frequency = {}
        for entry in self.detections_history:
            for cls_id, count in entry.get('object_classes', {}).items():
                frequency[f"Class_{cls_id}"] = frequency.get(f"Class_{cls_id}", 0) + count
        return frequency
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all detections"""
        if not self.detections_history:
            return 0.0
        return np.mean([e.get('average_confidence', 0) for e in self.detections_history])
    
    def export_to_csv(self, filepath: str):
        """Export detection history to CSV"""
        if not self.detections_history:
            logger.warning("No detection history to export")
            return
        
        df = pd.DataFrame(self.detections_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Detection history exported to {filepath}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.detections_history:
            return {}
        
        return {
            'total_detections': len(self.detections_history),
            'total_objects': sum(e.get('total_objects', 0) for e in self.detections_history),
            'average_confidence': self.get_average_confidence(),
            'object_frequency': self.get_object_frequency()
        }


class ModelManager:
    """Manage YOLOv8 models"""
    
    AVAILABLE_MODELS = {
        'YOLOv8 Nano': 'yolov8n.pt',
        'YOLOv8 Small': 'yolov8s.pt',
        'YOLOv8 Medium': 'yolov8m.pt',
        'YOLOv8 Large': 'yolov8l.pt',
        'YOLOv8 Extra Large': 'yolov8x.pt'
    }
    
    @staticmethod
    def get_available_models() -> Dict:
        """Get list of available models"""
        return ModelManager.AVAILABLE_MODELS
    
    @staticmethod
    def is_model_available(model_path: str) -> bool:
        """Check if model file exists"""
        return os.path.exists(model_path)
    
    @staticmethod
    def get_model_size(model_name: str) -> str:
        """Get model size information"""
        sizes = {
            'yolov8n.pt': '~3.2 MB',
            'yolov8s.pt': '~22.5 MB',
            'yolov8m.pt': '~49.0 MB',
            'yolov8l.pt': '~83.7 MB',
            'yolov8x.pt': '~135.0 MB'
        }
        return sizes.get(model_name, 'Unknown')


class ConfigManager:
    """Manage configuration settings"""
    
    DEFAULT_CONFIG = {
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_detections': 300,
        'model': 'yolov8n.pt',
        'device': 'cpu',
        'image_max_width': 1280,
        'image_max_height': 720
    }
    
    @staticmethod
    def load_config(filepath: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return ConfigManager.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config(filepath: str, config: Dict):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")


# Initialize analytics engine
analytics_engine = AnalyticsEngine()

logger.info("Utility modules loaded successfully")
