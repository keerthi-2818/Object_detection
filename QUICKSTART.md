# Quick Start Guide - Real-Time Object Detection Using YOLOv8

**Internship ID:** CITS2432

## 🚀 5-Minute Quick Start

### 1. Setup Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Required Directories
```bash
mkdir -p models outputs data logs config assets
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

**The application will open at:** `http://localhost:8501`

---

## 📸 Basic Usage Examples

### Using the Dashboard (Easiest)

1. **Load a Model:**
   - Open sidebar → Select "YOLOv8 Nano"
   - Choose "cpu" or "cuda"
   - Click "Load Model"

2. **Image Detection:**
   - Go to "Image Detection" tab
   - Click "Upload Image"
   - Select an image file
   - Click "Detect Objects"
   - Download result

3. **Video Detection:**
   - Go to "Video Detection" tab
   - Upload a video file
   - Click "Process Video"
   - Download processed video when done

4. **Live Webcam:**
   - Go to "Webcam Detection" tab
   - Click "Start Webcam Detection"
   - Press 'Q' to stop

### Using Python Code

**Example 1: Single Image**
```python
from detect_image import ImageDetector

detector = ImageDetector('yolov8n.pt', 'cpu')
image, stats = detector.detect_from_file(
    'image.jpg',
    output_path='outputs/result.jpg'
)

print(f"Found {stats['total_objects']} objects")
```

**Example 2: Webcam**
```python
from webcam_detection import WebcamDetector

detector = WebcamDetector('yolov8n.pt', 'cpu')
stats = detector.run()  # Press Q to exit

print(f"Detected {stats['total_objects']} objects in {stats['total_frames']} frames")
```

**Example 3: Video Processing**
```python
from detect_video import VideoDetector

detector = VideoDetector('yolov8n.pt', 'cpu')
stats = detector.detect_video(
    'video.mp4',
    output_path='outputs/video_result.mp4'
)

print(f"Average FPS: {stats['average_fps']:.2f}")
```

---

## 🎯 Model Selection Guide

Choose based on your needs:

| Use Case | Recommended | Speed | Accuracy |
|----------|-------------|-------|----------|
| Real-time Detection | **Nano** | ⚡⚡⚡ | Good |
| Fast Processing | **Small** | ⚡⚡ | Better |
| Balanced | **Medium** | ⚡ | Very Good |
| High Accuracy | **Large** | 🔄 | Excellent |
| Maximum Accuracy | **XLarge** | 🔄🔄 | Best |

---

## ⚙️ Configuration

### Quick Configuration Edit

Edit `config/settings.json`:
```json
{
  "confidence_threshold": 0.5,
  "model": "yolov8n.pt",
  "device": "cpu"
}
```

### Key Parameters

- **confidence_threshold:** 0.0-1.0 (higher = fewer detections)
- **iou_threshold:** 0.0-1.0 (NMS threshold)
- **device:** "cpu" or "cuda" (GPU acceleration)

---

## 🔧 Troubleshooting Quick Fixes

### GPU Not Working?
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU:
# Change device='cuda' to device='cpu'
```

### Slow Performance?
```bash
# Use smaller model
model = 'yolov8n.pt'  # Nano is fastest

# Or use GPU
device = 'cuda'
```

### Module Not Found?
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Webcam Not Working?
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different index (0, 1, 2, etc.)
```

---

## 📊 Understanding Results

### Detection Statistics
- **Objects Detected:** Total number of detected objects
- **Average Confidence:** Mean confidence score (0-1)
- **Max Confidence:** Highest confidence score
- **FPS:** Frames per second processed

### Bounding Boxes
- **Green Box:** Object boundary
- **Label:** "Object [ID]: [Confidence]"
- **Confidence:** Detection certainty (0-100%)

---

## 📁 File Organization

After first run, you'll have:
```
outputs/
├── detected_image_20231215_120530.jpg
└── video_detected_20231215_121200.mp4

data/
└── detection_history_20231215.csv

logs/
└── detection.log
```

---

## 🚨 Common Issues

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -r requirements.txt` |
| No detections | Lower confidence threshold |
| Very slow | Use smaller model or GPU |
| Webcam fails | Try different camera index |
| Out of memory | Use smaller image size |

---

## 📚 Next Steps

1. **Try the Dashboard:** `streamlit run app.py`
2. **Process Sample Images:** Place images in `assets/`
3. **Check Results:** View outputs in `outputs/` folder
4. **Export Data:** Download CSV from Analytics tab
5. **Read Full Docs:** See `README.md`

---

## 💡 Tips & Tricks

✅ **Batch Process Images:**
```python
from detect_image import ImageDetector

detector = ImageDetector()
results = detector.batch_detect('path/to/images/')
```

✅ **Export Results:**
```python
# Automatically logged and exportable from dashboard
# Or programmatically:
analytics.export_to_csv('data/results.csv')
```

✅ **Custom Models:**
Place your trained model in `models/` and load it:
```python
detector = ImageDetector('models/custom_model.pt')
```

---

## 🎓 Learning Resources

- [YOLOv8 Docs](https://docs.ultralytics.com)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Streamlit Docs](https://docs.streamlit.io)
- [PyTorch Docs](https://pytorch.org/docs)

---

## ✅ Verification Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Directories created (models, outputs, data, logs)
- [ ] Can run: `streamlit run app.py`
- [ ] Dashboard opens at `http://localhost:8501`
- [ ] Model loads without errors
- [ ] Can upload and detect objects

---

## 🎯 Success!

If you can see the Streamlit dashboard with model selection and detection tabs, you're ready to go! 

**Next:** Load a model and try detecting objects in an image.

---

**Internship ID:** CITS2432 | **Domain:** Artificial Intelligence & Computer Vision

For detailed information, see `README.md`
