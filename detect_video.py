"""
Video Object Detection Module
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, Callable
import time
from ultralytics import YOLO
from utils import DetectionUtils, AnalyticsEngine

logger = logging.getLogger(__name__)


class VideoDetector:
    """
    Video-based object detection using YOLOv8
    
    Features:
    - Process video frame-by-frame
    - Real-time object detection
    - FPS calculation and display
    - Save processed videos
    - Extract and log statistics
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        """
        Initialize VideoDetector
        
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
        
        logger.info(f"VideoDetector initialized with model: {model_path}")
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect_video(self, video_path: str,
                    confidence_threshold: float = 0.5,
                    iou_threshold: float = 0.45,
                    output_path: Optional[str] = None,
                    progress_callback: Optional[Callable] = None,
                    display_fps: bool = True) -> dict:
        """
        Detect objects in video file
        
        Args:
            video_path: Path to input video
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            output_path: Path to save processed video
            progress_callback: Callback function for progress updates
            display_fps: Display FPS on video
            
        Returns:
            Dictionary with detection statistics
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            # Setup video writer if output path provided
            out = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Processing statistics
            stats = {
                'total_frames': total_frames,
                'processed_frames': 0,
                'total_objects': 0,
                'average_fps': 0,
                'per_frame_stats': []
            }
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Resize frame if necessary
                frame = self.detection_utils.resize_image(frame)
                
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
                
                # Calculate FPS
                current_fps = frame_count / (time.time() - start_time)
                
                # Display FPS if requested
                if display_fps:
                    cv2.putText(annotated_frame, 
                               f"FPS: {current_fps:.2f}", 
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame,
                               f"Frame: {frame_count}/{total_frames}",
                               (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                
                # Save frame if output writer available
                if out:
                    out.write(annotated_frame)
                
                # Store per-frame statistics
                frame_data = {
                    'frame_number': frame_count,
                    'objects_detected': frame_stats['total_objects'],
                    'avg_confidence': frame_stats['average_confidence'],
                    'inference_time': inference_time,
                    'fps': current_fps
                }
                stats['per_frame_stats'].append(frame_data)
                
                # Update progress
                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress, frame_count, total_frames)
                
                logger.debug(f"Frame {frame_count}/{total_frames}: {frame_stats['total_objects']} objects")
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            stats['processed_frames'] = frame_count
            stats['average_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Video processing completed: {stats['processed_frames']} frames, "
                       f"Average FPS: {stats['average_fps']:.2f}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error during video detection: {str(e)}")
            return None
    
    def detect_from_webcam(self,
                          confidence_threshold: float = 0.5,
                          iou_threshold: float = 0.45,
                          display_fps: bool = True,
                          max_duration: Optional[int] = None,
                          progress_callback: Optional[Callable] = None) -> dict:
        """
        Real-time object detection from webcam
        
        Args:
            confidence_threshold: Minimum confidence score
            iou_threshold: IOU threshold for NMS
            display_fps: Display FPS on frame
            max_duration: Maximum duration in seconds (None for continuous)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with detection statistics
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
                'frames_with_objects': 0
            }
            
            frame_count = 0
            start_time = time.time()
            
            while True:
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
                    stats['frames_with_objects'] += 1
                
                # Calculate FPS
                current_fps = frame_count / (time.time() - start_time)
                
                # Display FPS
                if display_fps:
                    cv2.putText(annotated_frame,
                               f"FPS: {current_fps:.2f}",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame,
                               f"Objects: {frame_stats['total_objects']}",
                               (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Real-Time Object Detection', annotated_frame)
                
                # Check for exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check duration limit
                if max_duration:
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        break
                
                # Update progress
                if progress_callback:
                    progress_callback(frame_count)
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            elapsed_time = time.time() - start_time
            stats['total_frames'] = frame_count
            stats['average_fps'] = frame_count / elapsed_time if elapsed_time > 0 else 0
            stats['total_duration'] = elapsed_time
            
            logger.info(f"Webcam session completed: {stats['total_frames']} frames, "
                       f"Average FPS: {stats['average_fps']:.2f}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Error during webcam detection: {str(e)}")
            return None
    
    def get_analytics_summary(self) -> dict:
        """Get analytics summary"""
        return self.analytics.get_summary_stats()


def main():
    """Example usage"""
    # Initialize detector
    detector = VideoDetector(model_path='yolov8n.pt', device='cpu')
    
    # Example: Webcam detection
    print("Starting webcam detection... Press 'q' to exit")
    stats = detector.detect_from_webcam(
        confidence_threshold=0.5,
        display_fps=True
    )
    
    if stats:
        print(f"\nWebcam session completed!")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Total objects: {stats['total_objects']}")
        print(f"Average FPS: {stats['average_fps']:.2f}")


if __name__ == "__main__":
    main()
