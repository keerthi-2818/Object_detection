"""
Webcam Object Detection Module
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432
"""

import cv2
import numpy as np
import logging
from typing import Optional, Callable
import time
from ultralytics import YOLO
from utils import DetectionUtils, AnalyticsEngine

logger = logging.getLogger(__name__)


class WebcamDetector:
    """
    Real-time object detection from webcam
    
    Features:
    - Live webcam feed processing
    - Real-time bounding box visualization
    - FPS counter display
    - Object counting and tracking
    - Configurable detection parameters
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        """
        Initialize WebcamDetector
        
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
        self.running = False
        
        logger.info(f"WebcamDetector initialized with model: {model_path}")
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def run(self,
            confidence_threshold: float = 0.5,
            iou_threshold: float = 0.45,
            display_fps: bool = True,
            display_object_count: bool = True,
            max_duration: Optional[int] = None) -> dict:
        """
        Run live webcam detection
        
        Args:
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            display_fps: Display FPS counter
            display_object_count: Display object count
            max_duration: Maximum duration in seconds
            
        Returns:
            Dictionary with statistics
        """
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("Could not open webcam")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Webcam opened successfully")
            
            # Statistics
            stats = {
                'total_frames': 0,
                'total_objects': 0,
                'average_fps': 0,
                'total_duration': 0,
                'frames_with_detections': 0,
                'peak_fps': 0,
                'min_fps': float('inf')
            }
            
            frame_count = 0
            start_time = time.time()
            self.running = True
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                frame_start = time.time()
                results = self.model(
                    frame,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                inference_time = time.time() - frame_start
                
                # Draw bounding boxes
                annotated_frame = self.detection_utils.draw_bounding_boxes(
                    frame,
                    results[0],
                    confidence_threshold
                )
                
                # Get frame statistics
                frame_stats = self.detection_utils.get_detection_stats(results[0])
                stats['total_objects'] += frame_stats['total_objects']
                
                if frame_stats['total_objects'] > 0:
                    stats['frames_with_detections'] += 1
                
                # Calculate FPS
                current_fps = frame_count / (time.time() - start_time)
                stats['peak_fps'] = max(stats['peak_fps'], current_fps)
                stats['min_fps'] = min(stats['min_fps'], current_fps)
                
                # Prepare display text
                text_y = 30
                
                if display_fps:
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(annotated_frame,
                               fps_text,
                               (10, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                    text_y += 40
                
                if display_object_count:
                    count_text = f"Objects: {frame_stats['total_objects']}"
                    cv2.putText(annotated_frame,
                               count_text,
                               (10, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                    text_y += 40
                
                # Display confidence threshold
                conf_text = f"Confidence: {confidence_threshold:.2f}"
                cv2.putText(annotated_frame,
                           conf_text,
                           (10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Real-Time Object Detection - Press Q to Exit', annotated_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("User requested exit")
                    break
                
                # Check duration limit
                if max_duration:
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        break
            
            # Cleanup
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            stats['total_frames'] = frame_count
            stats['average_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
            stats['total_duration'] = elapsed_time
            
            # Add analytics
            self.analytics.add_detection(stats)
            
            logger.info(f"Webcam session completed:")
            logger.info(f"  Total frames: {stats['total_frames']}")
            logger.info(f"  Total objects: {stats['total_objects']}")
            logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
            logger.info(f"  Peak FPS: {stats['peak_fps']:.2f}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error during webcam detection: {str(e)}")
            self.running = False
            return None
    
    def stop(self):
        """Stop webcam detection"""
        self.running = False
        logger.info("Webcam detection stopped")
    
    def get_analytics_summary(self) -> dict:
        """Get analytics summary"""
        return self.analytics.get_summary_stats()


class SimpleWebcamDetector:
    """Simplified webcam detector for quick usage"""
    
    @staticmethod
    def detect(model_path: str = 'yolov8n.pt',
              confidence_threshold: float = 0.5,
              device: str = 'cpu') -> dict:
        """
        Quick webcam detection with defaults
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Confidence threshold
            device: Device to use
            
        Returns:
            Statistics dictionary
        """
        detector = WebcamDetector(model_path, device)
        return detector.run(confidence_threshold=confidence_threshold)


def main():
    """Example usage"""
    # Initialize detector
    detector = WebcamDetector(model_path='yolov8n.pt', device='cpu')
    
    # Run detection
    print("Starting real-time object detection...")
    print("Press 'q' to stop detection")
    print("-" * 50)
    
    stats = detector.run(
        confidence_threshold=0.5,
        display_fps=True,
        display_object_count=True
    )
    
    if stats:
        print("\n" + "="*50)
        print("DETECTION SESSION SUMMARY")
        print("="*50)
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Total Objects Detected: {stats['total_objects']}")
        print(f"Frames with Detections: {stats['frames_with_detections']}")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Peak FPS: {stats['peak_fps']:.2f}")
        print(f"Session Duration: {stats['total_duration']:.2f}s")
        print("="*50)


if __name__ == "__main__":
    main()
