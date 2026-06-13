"""
Streamlit Dashboard for Real-Time Object Detection
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432
Domain: Artificial Intelligence and Computer Vision
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Import custom modules
from detect_image import ImageDetector
from detect_video import VideoDetector
from webcam_detection import WebcamDetector
from utils import DetectionUtils, AnalyticsEngine, ModelManager, ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="YOLO Real-Time Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
if 'image_detector' not in st.session_state:
    st.session_state.image_detector = None

if 'video_detector' not in st.session_state:
    st.session_state.video_detector = None

if 'analytics' not in st.session_state:
    st.session_state.analytics = AnalyticsEngine()

if 'config' not in st.session_state:
    st.session_state.config = ConfigManager.load_config('config/settings.json')


def load_model(model_name: str, device: str = 'cpu'):
    """Load YOLOv8 model"""
    try:
        with st.spinner(f"Loading {model_name}..."):
            image_detector = ImageDetector(model_path=model_name, device=device)
            video_detector = VideoDetector(model_path=model_name, device=device)
            st.session_state.image_detector = image_detector
            st.session_state.video_detector = video_detector
            st.success(f"Model {model_name} loaded successfully!")
            logger.info(f"Model loaded: {model_name}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Model loading error: {str(e)}")


def detect_objects_in_image(image_path: str, confidence: float) -> tuple:
    """Detect objects in image"""
    if st.session_state.image_detector is None:
        st.error("Please load a model first!")
        return None, None
    
    try:
        annotated_image, stats = st.session_state.image_detector.detect_from_file(
            image_path,
            confidence_threshold=confidence
        )
        return annotated_image, stats
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        logger.error(f"Detection error: {str(e)}")
        return None, None


def detect_objects_in_video(video_path: str, confidence: float, progress_placeholder):
    """Detect objects in video"""
    if st.session_state.video_detector is None:
        st.error("Please load a model first!")
        return None
    
    try:
        output_path = DetectionUtils.create_output_path('outputs', 'video_detected', 'mp4')
        
        def progress_callback(progress, current, total):
            progress_placeholder.progress(progress / 100, text=f"Processing: {current}/{total} frames")
        
        stats = st.session_state.video_detector.detect_video(
            video_path,
            confidence_threshold=confidence,
            output_path=output_path,
            progress_callback=progress_callback
        )
        
        return stats, output_path
    except Exception as e:
        st.error(f"Error during video detection: {str(e)}")
        logger.error(f"Video detection error: {str(e)}")
        return None, None


# Header
st.title("🎯 Real-Time Object Detection Using YOLOv8")
st.markdown("""
    **Internship ID:** CITS2432 | **Domain:** Artificial Intelligence & Computer Vision
    
    A powerful real-time object detection system built with YOLOv8, OpenCV, and Streamlit.
    """)

# Sidebar - Configuration and Model Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    available_models = ModelManager.get_available_models()
    selected_model = st.selectbox(
        "Select YOLOv8 Model",
        list(available_models.keys()),
        help="Choose from different model sizes. Nano is fastest, Extra Large is most accurate."
    )
    
    # Device selection
    device = st.radio("Device", ["cpu", "cuda"], help="GPU (cuda) is faster but requires NVIDIA GPU")
    
    # Load model button
    if st.button("Load Model", key="load_model"):
        load_model(available_models[selected_model], device)
    
    st.divider()
    
    # Detection parameters
    st.subheader("Detection Parameters")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detected objects"
    )
    
    iou = st.slider(
        "IOU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )
    
    st.divider()
    
    # Information
    st.subheader("ℹ️ Model Information")
    if selected_model:
        model_name = available_models[selected_model]
        model_size = ModelManager.get_model_size(model_name)
        st.metric("Model Size", model_size)
        st.info("""
            **Model Sizes (from fastest to most accurate):**
            - 🚀 Nano (3.2 MB) - Best for real-time
            - ⚡ Small (22.5 MB) - Fast & accurate
            - ⚙️ Medium (49.0 MB) - Balanced
            - 🎯 Large (83.7 MB) - High accuracy
            - 🏆 XLarge (135.0 MB) - Best accuracy
        """)


# Main content - Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Image Detection",
    "🎬 Video Detection",
    "📹 Webcam Detection",
    "📊 Analytics",
    "📚 Documentation"
])

# =====================================================
# Tab 1: Image Detection
# =====================================================
with tab1:
    st.header("📸 Image Object Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_uploader"
        )
        
        if uploaded_image:
            # Save uploaded file temporarily
            temp_image_path = f"temp_{uploaded_image.name}"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Display original image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Detect button
            if st.button("🔍 Detect Objects", key="detect_image_btn"):
                if st.session_state.image_detector is None:
                    st.error("Please load a model first!")
                else:
                    with st.spinner("Processing image..."):
                        annotated_image, stats = detect_objects_in_image(
                            temp_image_path,
                            confidence
                        )
                    
                    if annotated_image is not None:
                        # Save result
                        output_path = DetectionUtils.create_output_path(
                            'outputs',
                            'detected_image',
                            'jpg'
                        )
                        cv2.imwrite(output_path, annotated_image)
                        
                        # Display results in second column
                        with col2:
                            st.subheader("Detection Results")
                            detected_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            st.image(detected_image_rgb, caption="Detected Objects", use_column_width=True)
                            
                            # Statistics
                            if stats:
                                st.success(f"✅ Detection Complete!")
                                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                                
                                with col_metrics1:
                                    st.metric("Objects Detected", stats['total_objects'])
                                
                                with col_metrics2:
                                    st.metric("Avg Confidence", f"{stats['average_confidence']:.3f}")
                                
                                with col_metrics3:
                                    st.metric("Max Confidence", f"{stats['max_confidence']:.3f}")
                                
                                # Object class distribution
                                if stats['object_classes']:
                                    st.subheader("Object Classes")
                                    class_data = pd.DataFrame(
                                        list(stats['object_classes'].items()),
                                        columns=['Class ID', 'Count']
                                    )
                                    st.table(class_data)
                            
                            # Download button
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Detected Image",
                                    data=f.read(),
                                    file_name=os.path.basename(output_path),
                                    mime="image/jpeg"
                                )
            
            # Cleanup
            try:
                os.remove(temp_image_path)
            except:
                pass


# =====================================================
# Tab 2: Video Detection
# =====================================================
with tab2:
    st.header("🎬 Video Object Detection")
    
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )
    
    if uploaded_video:
        # Save uploaded file temporarily
        temp_video_path = f"temp_{uploaded_video.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.info(f"Video loaded: {uploaded_video.name}")
        
        # Detect button
        if st.button("🎯 Process Video", key="detect_video_btn"):
            if st.session_state.video_detector is None:
                st.error("Please load a model first!")
            else:
                progress_placeholder = st.empty()
                
                with st.spinner("Processing video..."):
                    stats, output_path = detect_objects_in_video(
                        temp_video_path,
                        confidence,
                        progress_placeholder
                    )
                
                if stats:
                    st.success("✅ Video Processing Complete!")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Frames", stats['total_frames'])
                    
                    with col2:
                        st.metric("Objects Detected", stats['total_objects'])
                    
                    with col3:
                        st.metric("Average FPS", f"{stats['average_fps']:.2f}")
                    
                    with col4:
                        avg_objects_per_frame = stats['total_objects'] / stats['processed_frames'] if stats['processed_frames'] > 0 else 0
                        st.metric("Avg Objects/Frame", f"{avg_objects_per_frame:.2f}")
                    
                    # Frame-by-frame statistics
                    if stats['per_frame_stats']:
                        st.subheader("Frame-by-Frame Analysis")
                        
                        df = pd.DataFrame(stats['per_frame_stats'])
                        
                        # Create charts
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(df['frame_number'], df['objects_detected'], color='#667eea', linewidth=2)
                            ax.fill_between(df['frame_number'], df['objects_detected'], alpha=0.3, color='#667eea')
                            ax.set_xlabel('Frame Number')
                            ax.set_ylabel('Objects Detected')
                            ax.set_title('Objects Detected per Frame')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        with chart_col2:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(df['frame_number'], df['fps'], color='#764ba2', linewidth=2)
                            ax.fill_between(df['frame_number'], df['fps'], alpha=0.3, color='#764ba2')
                            ax.set_xlabel('Frame Number')
                            ax.set_ylabel('FPS')
                            ax.set_title('FPS per Frame')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    # Download button
                    if output_path and os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=f.read(),
                                file_name=os.path.basename(output_path),
                                mime="video/mp4"
                            )
        
        # Cleanup
        try:
            os.remove(temp_video_path)
        except:
            pass


# =====================================================
# Tab 3: Webcam Detection
# =====================================================
with tab3:
    st.header("📹 Real-Time Webcam Detection")
    
    st.info("Click 'Start Detection' to begin webcam object detection. Press 'Q' in the detection window to exit.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        webcam_confidence = st.slider(
            "Webcam Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="webcam_confidence"
        )
    
    with col2:
        max_duration = st.number_input(
            "Max Duration (seconds)",
            min_value=0,
            value=0,
            help="0 = continuous until exit"
        )
    
    if st.button("🎥 Start Webcam Detection", key="webcam_btn"):
        if st.session_state.image_detector is None:
            st.error("Please load a model first!")
        else:
            try:
                detector = WebcamDetector(
                    model_path=list(ModelManager.get_available_models().values())[0],
                    device=device
                )
                
                status_placeholder = st.empty()
                status_placeholder.info("Starting webcam... Press 'Q' to stop.")
                
                stats = detector.run(
                    confidence_threshold=webcam_confidence,
                    display_fps=True,
                    display_object_count=True,
                    max_duration=max_duration if max_duration > 0 else None
                )
                
                if stats:
                    status_placeholder.success("✅ Webcam session completed!")
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Frames", stats['total_frames'])
                    
                    with col2:
                        st.metric("Objects Detected", stats['total_objects'])
                    
                    with col3:
                        st.metric("Average FPS", f"{stats['average_fps']:.2f}")
                    
                    with col4:
                        st.metric("Session Duration", f"{stats['total_duration']:.2f}s")
                    
                    # Additional metrics
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        st.metric("Frames with Objects", stats['frames_with_detections'])
                    
                    with col6:
                        detection_rate = (stats['frames_with_detections'] / stats['total_frames'] * 100) if stats['total_frames'] > 0 else 0
                        st.metric("Detection Rate", f"{detection_rate:.1f}%")
                    
                    # Add to analytics
                    st.session_state.analytics.add_detection(stats)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Webcam error: {str(e)}")


# =====================================================
# Tab 4: Analytics
# =====================================================
with tab4:
    st.header("📊 Analytics & Statistics")
    
    # Summary statistics
    summary = st.session_state.analytics.get_summary_stats()
    
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", summary.get('total_detections', 0))
        
        with col2:
            st.metric("Total Objects", summary.get('total_objects', 0))
        
        with col3:
            st.metric("Avg Confidence", f"{summary.get('average_confidence', 0):.3f}")
        
        with col4:
            st.metric("Object Types", len(summary.get('object_frequency', {})))
        
        # Object frequency
        if summary.get('object_frequency'):
            st.subheader("Object Frequency Distribution")
            freq_data = pd.DataFrame(
                list(summary['object_frequency'].items()),
                columns=['Object Class', 'Count']
            )
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(freq_data['Object Class'], freq_data['Count'], color='#667eea')
            ax.set_xlabel('Object Class')
            ax.set_ylabel('Count')
            ax.set_title('Detection Frequency by Object Class')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            
            st.table(freq_data)
    else:
        st.info("No detection history yet. Perform some detections to see analytics.")
    
    # Export history
    st.subheader("📥 Export Detection History")
    
    if st.button("📊 Export to CSV"):
        csv_path = DetectionUtils.create_output_path('data', 'detection_history', 'csv')
        st.session_state.analytics.export_to_csv(csv_path)
        
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Detection History",
                    data=f.read(),
                    file_name=os.path.basename(csv_path),
                    mime="text/csv"
                )


# =====================================================
# Tab 5: Documentation
# =====================================================
with tab5:
    st.header("📚 Project Documentation")
    
    tab_docs1, tab_docs2, tab_docs3 = st.tabs(["About", "Features", "Usage Guide"])
    
    with tab_docs1:
        st.subheader("About This Project")
        st.markdown("""
            ### Real-Time Object Detection Using YOLOv8
            
            **Internship ID:** CITS2432  
            **Domain:** Artificial Intelligence & Computer Vision  
            **Programming Language:** Python  
            **Framework:** YOLOv8 (Ultralytics)
            
            #### Overview
            This project implements a comprehensive real-time object detection system using YOLOv8,
            a state-of-the-art deep learning model for object detection. The system supports
            detection from images, videos, and live webcam feeds.
            
            #### Key Technologies
            - **YOLOv8:** Ultralytics YOLO object detection framework
            - **OpenCV:** Computer vision and image processing
            - **Streamlit:** Interactive web dashboard
            - **NumPy & Pandas:** Numerical and data processing
            - **Matplotlib:** Data visualization
            
            #### Architecture
            The project is built with a modular architecture:
            - **utils.py:** Core utilities and analytics
            - **detect_image.py:** Image detection module
            - **detect_video.py:** Video detection module
            - **webcam_detection.py:** Real-time webcam detection
            - **app.py:** Streamlit dashboard interface
        """)
    
    with tab_docs2:
        st.subheader("Key Features")
        st.markdown("""
            ✅ **Image Detection**
            - Upload and process single images
            - Detect multiple objects with bounding boxes
            - Display confidence scores
            - Download processed images
            
            ✅ **Video Detection**
            - Process video frame-by-frame
            - Real-time object detection
            - FPS calculation and display
            - Save processed videos
            - Frame-by-frame statistics
            
            ✅ **Webcam Detection**
            - Live webcam object detection
            - Real-time bounding box visualization
            - FPS counter display
            - Object counting and statistics
            
            ✅ **Analytics & Statistics**
            - Track detection history
            - Calculate detection statistics
            - Generate frequency charts
            - Export results to CSV
            
            ✅ **Model Management**
            - Support for 5 YOLOv8 model sizes
            - CPU and GPU device support
            - Dynamic model loading
            - Configurable detection parameters
        """)
    
    with tab_docs3:
        st.subheader("Usage Guide")
        st.markdown("""
            ### Getting Started
            
            1. **Load a Model**
               - Select desired YOLOv8 model from sidebar
               - Choose CPU or GPU device
               - Click "Load Model" button
            
            2. **Image Detection**
               - Go to "Image Detection" tab
               - Upload an image
               - Adjust confidence threshold if needed
               - Click "Detect Objects"
               - Download processed image
            
            3. **Video Detection**
               - Go to "Video Detection" tab
               - Upload a video file
               - Click "Process Video"
               - Monitor progress and download results
            
            4. **Webcam Detection**
               - Go to "Webcam Detection" tab
               - Set confidence threshold
               - Click "Start Webcam Detection"
               - Press 'Q' in webcam window to stop
            
            5. **View Analytics**
               - Go to "Analytics" tab
               - View detection statistics
               - Export history to CSV
            
            ### Tips & Tricks
            - Use smaller models (Nano/Small) for faster inference
            - Use larger models (Large/XLarge) for better accuracy
            - Adjust confidence threshold based on your needs
            - GPU acceleration significantly improves performance
            - Lower confidence thresholds detect more objects
        """)


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Real-Time Object Detection Using YOLOv8 | Internship ID: CITS2432</p>
        <p>Built with ❤️ using Streamlit and YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)

logger.info("Streamlit app running successfully")
