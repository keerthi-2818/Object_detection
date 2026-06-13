"""
Image Object Detection Module
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple
from ultralytics import YOLO
from utils import DetectionUtils, AnalyticsEngine, ModelManager

logger = logging.getLogger(__name__)


class ImageDetector:
    """
    Image-based object detection using YOLOv8
    
    Features:
    - Load and process single images
    - Detect multiple objects
    - Draw bounding boxes with confidence scores
    - Save detection results
    - Extract and log statistics
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        """
        Initialize ImageDetector
        
        Args:
            model_path: Path to YOLOv8 model file
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.load_model()
        self.detection_utils = DetectionUtils()
        self.analytics = AnalyticsEngine()
        
        logger.info(f"ImageDetector initialized with model: {model_path}")
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect(self, image_path: str, 
               confidence_threshold: float = 0.5,
               iou_threshold: float = 0.45) -> Tuple[np.ndarray, dict, dict]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            
        Returns:
            Tuple of (annotated_image, raw_detections, statistics)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None, None, None
            
            original_image = image.copy()
            
            # Resize if necessary
            image = self.detection_utils.resize_image(image)
            
            # Run inference
            results = self.model(
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Draw bounding boxes
            annotated_image = self.detection_utils.draw_bounding_boxes(
                image, 
                results[0],
                confidence_threshold
            )
            
            # Get statistics
            stats = self.detection_utils.get_detection_stats(results[0])
            
            # Log detection
            self.analytics.add_detection(stats)
            self.detection_utils.log_detection(image_path, results[0], stats)
            
            logger.info(f"Detection completed: {image_path} - Found {stats['total_objects']} objects")
            
            return annotated_image, results[0], stats
        
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return None, None, None
    
    def detect_from_file(self, image_path: str,
                        confidence_threshold: float = 0.5,
                        iou_threshold: float = 0.45,
                        output_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Detect objects and optionally save result
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            output_path: Path to save annotated image
            
        Returns:
            Tuple of (annotated_image, statistics)
        """
        annotated_image, detections, stats = self.detect(
            image_path,
            confidence_threshold,
            iou_threshold
        )
        
        if annotated_image is None:
            return None, None
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_image)
            logger.info(f"Annotated image saved to {output_path}")
        
        return annotated_image, stats
    
    def batch_detect(self, image_dir: str,
                    confidence_threshold: float = 0.5,
                    output_dir: str = 'outputs') -> dict:
        """
        Detect objects in multiple images
        
        Args:
            image_dir: Directory containing images
            confidence_threshold: Minimum confidence score
            output_dir: Directory to save results
            
        Returns:
            Dictionary with batch results
        """
        results = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'total_objects_detected': 0,
            'details': []
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        results['total_images'] = len(image_files)
        
        for idx, image_file in enumerate(image_files, 1):
            try:
                image_path = os.path.join(image_dir, image_file)
                
                # Create output filename
                name, ext = os.path.splitext(image_file)
                output_path = os.path.join(output_dir, f"{name}_detected.jpg")
                
                # Run detection
                annotated_image, stats = self.detect_from_file(
                    image_path,
                    confidence_threshold,
                    output_path
                )
                
                if annotated_image is not None:
                    results['processed'] += 1
                    results['total_objects_detected'] += stats['total_objects']
                    results['details'].append({
                        'file': image_file,
                        'objects': stats['total_objects'],
                        'avg_confidence': stats['average_confidence']
                    })
                    logger.info(f"[{idx}/{results['total_images']}] Processed: {image_file}")
                else:
                    results['failed'] += 1
                    logger.warning(f"[{idx}/{results['total_images']}] Failed: {image_file}")
            
            except Exception as e:
                results['failed'] += 1
                logger.error(f"Error processing {image_file}: {str(e)}")
        
        logger.info(f"Batch detection completed: {results['processed']}/{results['total_images']} successful")
        return results
    
    def get_analytics_summary(self) -> dict:
        """Get analytics summary"""
        return self.analytics.get_summary_stats()


def main():
    """Example usage"""
    # Initialize detector
    detector = ImageDetector(model_path='yolov8n.pt', device='cpu')
    
    # Example: Single image detection
    test_image = 'assets/sample_image.jpg'
    if os.path.exists(test_image):
        output_path = DetectionUtils.create_output_path('outputs', 'detected_image', 'jpg')
        annotated_image, stats = detector.detect_from_file(
            test_image,
            output_path=output_path
        )
        
        if annotated_image is not None:
            print(f"Detection completed!")
            print(f"Objects detected: {stats['total_objects']}")
            print(f"Average confidence: {stats['average_confidence']:.2f}")
    else:
        print(f"Test image not found: {test_image}")


if __name__ == "__main__":
    main()
