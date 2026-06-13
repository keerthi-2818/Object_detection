# Real-Time Object Detection Using YOLOv8

**Internship ID:** CITS2432  
**Domain:** Artificial Intelligence and Computer Vision  
**Programming Language:** Python  
**Framework:** YOLOv8 (Ultralytics)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Modules Documentation](#modules-documentation)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 📖 Overview

This project implements a comprehensive **Real-Time Object Detection System** using YOLOv8, a state-of-the-art deep learning model for object detection. The system can detect and identify multiple objects from:

- 📸 **Static Images**
- 🎬 **Video Files**
- 📹 **Live Webcam Feeds**

The project features a professional **Streamlit Dashboard** for easy interaction, supporting multiple YOLOv8 model sizes (Nano to XLarge), and provides comprehensive analytics and statistics.

### Key Highlights

✅ **Real-Time Detection** - Process video feeds and webcam at high FPS  
✅ **Multi-Source Support** - Images, videos, and live webcam  
✅ **Flexible Models** - 5 YOLOv8 variants to choose from  
✅ **Production-Ready** - Well-documented, tested, and optimized code  
✅ **Interactive Dashboard** - Streamlit web interface for easy use  
✅ **Analytics Engine** - Track and export detection statistics  
✅ **GPU Support** - CUDA acceleration for faster inference  

---

## ✨ Features

### 1. Image Object Detection
- ✅ Upload single or batch images
- ✅ Detect multiple objects with high accuracy
- ✅ Draw bounding boxes with labels
- ✅ Display confidence scores for each detection
- ✅ Save annotated images
- ✅ Batch processing support

### 2. Video Object Detection
- ✅ Upload and process video files
- ✅ Frame-by-frame object detection
- ✅ Real-time detection during processing
- ✅ FPS calculation and display
- ✅ Save processed videos with detections
- ✅ Per-frame statistics tracking

### 3. Webcam Detection
- ✅ Live object detection from webcam
- ✅ Real-time bounding box visualization
- ✅ FPS counter display
- ✅ Object counting and frequency tracking
- ✅ Configurable detection parameters
- ✅ Session statistics collection

### 4. YOLOv8 Integration
- ✅ Use pretrained YOLOv8 models (Nano to XLarge)
- ✅ Support for custom trained models
- ✅ Dynamic model loading
- ✅ Device selection (CPU/GPU)
- ✅ Configurable inference parameters

### 5. Professional Streamlit Dashboard
- ✅ Modern and responsive UI design
- ✅ Sidebar navigation and controls
- ✅ Multi-tab interface
- ✅ Real-time parameter adjustment
- ✅ Results visualization
- ✅ Download functionality

### 6. Analytics & Reporting
- ✅ Count detected objects per session
- ✅ Object frequency statistics
- ✅ Confidence score analysis
- ✅ Generate charts and summaries
- ✅ Export results to CSV
- ✅ Detection history logging

---

## 📁 Project Structure

```
Real_Time_Object_Detection/
│
├── app.py                      # Main Streamlit dashboard application
├── detect_image.py             # Image detection module
├── detect_video.py             # Video detection module
├── webcam_detection.py         # Webcam detection module
├── utils.py                    # Utility functions and classes
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── config/
│   └── settings.json           # Configuration settings
│
├── models/
│   └── yolov8n.pt             # YOLOv8 Nano model (auto-downloaded)
│
├── outputs/
│   ├── detected_images/        # Saved detection results (images)
│   ├── detected_videos/        # Saved detection results (videos)
│   └── ...
│
├── data/
│   └── detection_history.csv   # Detection history logs
│
├── logs/
│   └── detection.log           # Application logs
│
└── assets/
    └── sample_images/          # Sample images for testing

```

---

## 🔧 Prerequisites

### System Requirements

- **OS:** Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA support)

### Software Requirements

- **Git** - For cloning the repository
- **Python 3.8+** - Programming language
- **pip** - Python package manager
- **Virtual Environment** - For dependency isolation

---

## 📦 Installation Guide

### Step 1: Clone or Create Project Directory

```bash
# Create project directory
mkdir Real_Time_Object_Detection
cd Real_Time_Object_Detection

# Or clone if you have a Git repository
# git clone <repository-url>
```

### Step 2: Create and Activate Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Download YOLOv8 Models (Optional)

Models are automatically downloaded on first use. To pre-download specific models:

```bash
# This will download the model files
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 5: Create Necessary Directories

```bash
# Create required directories
mkdir models outputs data logs config assets
```

### Step 6: Run the Application

**Using Streamlit Dashboard (Recommended):**
```bash
streamlit run app.py
```

This will open the Streamlit application in your default browser at `http://localhost:8501`

---

## 🚀 Usage

### Using Streamlit Dashboard (Recommended)

1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Load a Model:**
   - Select desired YOLOv8 model from sidebar (Nano to XLarge)
   - Choose CPU or GPU device
   - Click "Load Model"

3. **Image Detection:**
   - Go to "Image Detection" tab
   - Upload an image
   - Adjust confidence threshold if needed
   - Click "Detect Objects"
   - View results and download annotated image

4. **Video Detection:**
   - Go to "Video Detection" tab
   - Upload a video file
   - Click "Process Video"
   - Monitor progress and download processed video

5. **Webcam Detection:**
   - Go to "Webcam Detection" tab
   - Click "Start Webcam Detection"
   - Press 'Q' in the detection window to stop

6. **View Analytics:**
   - Go to "Analytics" tab
   - View detection statistics and charts
   - Export history to CSV

### Using Python Scripts Directly

#### Image Detection Example:
```python
from detect_image import ImageDetector
from utils import DetectionUtils

# Initialize detector
detector = ImageDetector(model_path='yolov8n.pt', device='cpu')

# Detect objects
annotated_image, stats = detector.detect_from_file(
    'path/to/image.jpg',
    confidence_threshold=0.5,
    output_path='outputs/result.jpg'
)

# Print statistics
print(f"Objects detected: {stats['total_objects']}")
print(f"Average confidence: {stats['average_confidence']:.2f}")
```

#### Video Detection Example:
```python
from detect_video import VideoDetector

# Initialize detector
detector = VideoDetector(model_path='yolov8n.pt', device='cpu')

# Process video
stats = detector.detect_video(
    'path/to/video.mp4',
    confidence_threshold=0.5,
    output_path='outputs/video_detected.mp4'
)

print(f"Total objects: {stats['total_objects']}")
print(f"Average FPS: {stats['average_fps']:.2f}")
```

#### Webcam Detection Example:
```python
from webcam_detection import WebcamDetector

# Initialize detector
detector = WebcamDetector(model_path='yolov8n.pt', device='cpu')

# Run webcam detection
stats = detector.run(
    confidence_threshold=0.5,
    display_fps=True
)

print(f"Session statistics: {stats}")
```

---

## 🏗️ Project Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│  ┌──────────┬──────────┬──────────┬───────────────────┐ │
│  │  Image   │  Video   │ Webcam   │    Analytics      │ │
│  │Detection │Detection │Detection │  & Documentation │ │
│  └──────────┴──────────┴──────────┴───────────────────┘ │
└────────────────────────────────────┬────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
        ┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
        │ ImageDetector  │  │ VideoDetector  │  │WebcamDetector  │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   YOLOv8 Model (YOLO)  │
                        │  - yolov8n.pt (Nano)   │
                        │  - yolov8s.pt (Small)  │
                        │  - yolov8m.pt (Medium) │
                        │  - yolov8l.pt (Large)  │
                        │  - yolov8x.pt (XLarge) │
                        └────────────┬────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────▼─────┐            ┌────────▼────────┐         ┌────────▼─────┐
   │  OpenCV  │            │    NumPy/SciPy  │         │  Matplotlib  │
   │ (CV2)    │            │   Image Process  │         │ (Plotting)   │
   └──────────┘            └──────────────────┘         └──────────────┘
```

### Data Flow

```
Input Source          Detection Pipeline         Output & Storage
────────────          ──────────────────         ────────────────

📸 Image      ──→  Load Image  ──→  YOLOv8   ──→  Bounding Boxes  ──→ 💾 Save
              │     Preprocess  │    Inference │  Confidence Scores   │  Results
              │     Resize      │               │  Statistics          │
              │                 │               │                      │
🎬 Video      ──→  Frame        ──→  YOLOv8   ──→  Annotated Video  ──→ 📊 Analytics
              │     Extraction  │    Inference │  FPS Metrics         │  (CSV/JSON)
              │     Processing  │               │  Per-frame Stats     │
              │                 │               │                      │
📹 Webcam     ──→  Capture      ──→  YOLOv8   ──→  Real-time Display ──→ 📈 Charts
              │     Live Feed   │    Inference │  FPS Counter         │  & Stats
              │     Buffering   │               │  Object Count        │
```

---

## 📚 Modules Documentation

### utils.py

Core utility functions and classes for object detection.

**Key Classes:**

- **`DetectionUtils`** - Drawing, statistics, and processing utilities
  - `draw_bounding_boxes()` - Draw detection boxes on images
  - `get_detection_stats()` - Extract detection statistics
  - `calculate_fps()` - Calculate frames per second
  - `resize_image()` - Resize images while maintaining aspect ratio
  - `log_detection()` - Log detection results

- **`AnalyticsEngine`** - Analytics and statistics tracking
  - `add_detection()` - Add detection to history
  - `get_object_frequency()` - Get object frequency distribution
  - `get_average_confidence()` - Calculate average confidence
  - `export_to_csv()` - Export history to CSV

- **`ModelManager`** - YOLOv8 model management
  - `get_available_models()` - List available models
  - `is_model_available()` - Check if model exists
  - `get_model_size()` - Get model file size information

- **`ConfigManager`** - Configuration management
  - `load_config()` - Load configuration from JSON
  - `save_config()` - Save configuration to JSON

### detect_image.py

Image-based object detection module.

**Key Class:**

- **`ImageDetector`** - Image detection operations
  - `__init__()` - Initialize detector with model
  - `load_model()` - Load YOLOv8 model
  - `detect()` - Detect objects in image
  - `detect_from_file()` - Detect and save results
  - `batch_detect()` - Process multiple images
  - `get_analytics_summary()` - Get analytics

### detect_video.py

Video-based object detection module.

**Key Class:**

- **`VideoDetector`** - Video detection operations
  - `__init__()` - Initialize detector
  - `load_model()` - Load YOLOv8 model
  - `detect_video()` - Process video file
  - `detect_from_webcam()` - Webcam detection
  - `get_analytics_summary()` - Get analytics

### webcam_detection.py

Real-time webcam detection module.

**Key Classes:**

- **`WebcamDetector`** - Advanced webcam detection
  - `run()` - Run live detection with statistics
  - `stop()` - Stop detection gracefully

- **`SimpleWebcamDetector`** - Simplified interface
  - `detect()` - Quick webcam detection

### app.py

Streamlit dashboard application.

**Features:**
- Multi-tab interface
- Real-time parameter adjustment
- Results visualization
- Analytics and reporting
- Model management
- File upload and download

---

## ⚙️ Configuration

### Configuration File Structure

Create `config/settings.json`:

```json
{
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "max_detections": 300,
  "model": "yolov8n.pt",
  "device": "cpu",
  "image_max_width": 1280,
  "image_max_height": 720
}
```

### Environment Variables (Optional)

Create `.env` file:

```
YOLO_MODEL_PATH=models/
DETECTION_OUTPUT_DIR=outputs/
LOG_LEVEL=INFO
GPU_ENABLED=false
```

---

## 📊 Evaluation Metrics

### YOLOv8 Performance Metrics

The system evaluates detections using standard object detection metrics:

#### 1. **Precision**
- Ratio of correct detections to total detections
- Formula: `TP / (TP + FP)`
- Higher is better (closer to 1.0)

#### 2. **Recall**
- Ratio of correct detections to all objects
- Formula: `TP / (TP + FN)`
- Higher is better (closer to 1.0)

#### 3. **mAP (Mean Average Precision)**
- Average precision across all object classes
- Computed using precision-recall curve
- Standard metric for YOLO evaluation

#### 4. **Confidence Score**
- Model's confidence in detection (0-1)
- Higher indicates more certain detection
- Configurable threshold for filtering

#### 5. **FPS (Frames Per Second)**
- Real-time detection speed
- Varies by model size and hardware
- **YOLOv8 Nano:** ~100+ FPS (GPU)
- **YOLOv8 XLarge:** ~20+ FPS (GPU)

### Model Comparison

| Model | Size | Speed (GPU) | Accuracy |
|-------|------|-----------|----------|
| Nano | 3.2 MB | ~130 FPS | Good |
| Small | 22.5 MB | ~100 FPS | Better |
| Medium | 49.0 MB | ~70 FPS | Very Good |
| Large | 83.7 MB | ~40 FPS | Excellent |
| XLarge | 135.0 MB | ~20 FPS | Best |

---

## 📸 Screenshots

### 1. Dashboard Home Page
```
[Streamlit Interface showing YOLO Object Detection Dashboard]
- Header with internship ID
- Multi-tab interface
- Configuration sidebar
- Real-time results display
```

### 2. Image Detection Tab
```
[Before/After Detection]
Left: Original image upload
Right: Detection results with:
  - Bounding boxes
  - Confidence scores
  - Object counts
  - Download button
```

### 3. Video Detection Tab
```
[Video Processing Interface]
- Video upload area
- Processing progress bar
- Per-frame statistics
- FPS chart
- Download processed video
```

### 4. Webcam Detection Tab
```
[Live Webcam Interface]
- Real-time video stream
- FPS counter overlay
- Object count display
- Session statistics
```

### 5. Analytics Dashboard
```
[Analytics Page]
- Detection statistics cards
- Object frequency bar chart
- Trend analysis
- Export to CSV button
- Detection history table
```

---

## 🚀 Future Enhancements

### Phase 2 Features

1. **Custom Model Training**
   - Train YOLOv8 on custom datasets
   - Dataset configuration with data.yaml
   - Model fine-tuning capabilities
   - Weight saving and versioning

2. **Advanced Analytics**
   - Real-time dashboard metrics
   - Temporal analysis (time-series)
   - Multi-session comparison
   - Performance benchmarking

3. **Model Optimization**
   - Model quantization (INT8)
   - Model pruning
   - Knowledge distillation
   - ONNX export support

4. **Enhanced Features**
   - Object tracking (DeepSORT)
   - Multi-object tracking (MOT)
   - Pose estimation
   - Instance segmentation

5. **API & Integration**
   - REST API for detection
   - WebSocket for real-time updates
   - Docker containerization
   - Cloud deployment (AWS, Azure, GCP)

6. **Performance Features**
   - Batch inference optimization
   - Model ensemble support
   - Edge device deployment
   - Mobile app version

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found
**Error:** `FileNotFoundError: yolov8n.pt not found`

**Solution:**
```bash
# Manually download the model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### 2. Out of Memory (OOM)
**Error:** `CUDA out of memory` or `MemoryError`

**Solution:**
- Use smaller model (Nano instead of XLarge)
- Reduce image/frame size
- Run on CPU instead of GPU
- Increase system RAM

#### 3. Webcam Not Working
**Error:** `Could not open webcam` or `VideoCapture(0) failed`

**Solution:**
```bash
# Check if camera is available
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different camera indices
# Change VideoCapture(0) to VideoCapture(1), VideoCapture(2), etc.
```

#### 4. Slow Performance
**Issue:** FPS is very low

**Solutions:**
- Use GPU acceleration (`device='cuda'`)
- Use smaller model (Nano/Small)
- Reduce image resolution
- Close other applications

#### 5. Import Errors
**Error:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install specific package
pip install ultralytics
```

#### 6. CUDA Issues
**Error:** `CUDA not available` or `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
device = 'cpu'  # Set in code or config
```

---

## 📖 References

### Documentation
- [YOLOv8 Official Documentation](https://docs.ultralytics.com)
- [OpenCV Documentation](https://docs.opencv.org)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Documentation](https://pytorch.org/docs)

### Research Papers
- [YOLOv8: A Vision Transformer based Object Detection](https://arxiv.org/abs/2304.01143)
- [You Only Look Once (Original YOLO)](https://arxiv.org/abs/1506.02640)
- [YOLOv3 & YOLOv4 Papers](https://arxiv.org/abs/2004.10934)

### Related Projects
- [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [OpenCV GitHub](https://github.com/opencv/opencv)
- [Streamlit GitHub](https://github.com/streamlit/streamlit)

---

## 📝 License and Attribution

This project is built using open-source technologies:
- **YOLOv8** by Ultralytics (AGPL-3.0)
- **OpenCV** (Apache 2.0)
- **Streamlit** (Apache 2.0)
- **PyTorch** (BSD)

---

## 👨‍💼 Author & Contact

**Project:** Real-Time Object Detection Using YOLOv8  
**Internship ID:** CITS2432  
**Domain:** Artificial Intelligence & Computer Vision  

### Getting Help
- Check the troubleshooting section
- Review the documentation
- Check Ultralytics documentation
- Review error logs in `logs/detection.log`

---

## 📋 Version History

### v1.0.0 (Current)
- ✅ Image detection functionality
- ✅ Video detection functionality
- ✅ Webcam real-time detection
- ✅ Streamlit dashboard
- ✅ Analytics and reporting
- ✅ Multi-model support
- ✅ CSV export functionality
- ✅ Comprehensive documentation

### Future Versions
- v1.1.0 - Custom model training
- v1.2.0 - Advanced analytics
- v1.3.0 - REST API
- v2.0.0 - Mobile app

---

## ✅ Checklist for Success

- [x] Complete project structure
- [x] All detection modules implemented
- [x] Streamlit dashboard functional
- [x] Analytics engine working
- [x] Comprehensive documentation
- [x] Model management system
- [x] Error handling and logging
- [x] CSV export functionality
- [x] Production-ready code
- [x] Professional presentation

---

## 🎓 Educational Value

This project demonstrates:
- Deep Learning fundamentals (YOLOv8)
- Computer Vision techniques
- Object Detection methodologies
- Real-time processing
- Web application development (Streamlit)
- Data analytics and visualization
- Software engineering best practices
- Code documentation and maintenance

---

**Happy Detecting! 🎯**

For more information, visit the [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)

---

*Last Updated: December 2024*  
*Internship ID: CITS2432*  
*Domain: Artificial Intelligence & Computer Vision*
