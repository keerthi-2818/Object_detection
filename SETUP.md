# SETUP INSTRUCTIONS

**Real-Time Object Detection Using YOLOv8**  
**Internship ID:** CITS2432

---

## ⚡ Quick Setup (Easiest Way)

### Windows Users
```bash
# Just double-click this file:
startup.bat
```

### macOS/Linux Users
```bash
# Make the script executable and run it:
chmod +x startup.sh
./startup.sh
```

These scripts will automatically:
1. Create a virtual environment
2. Install all dependencies
3. Create required directories
4. Start the Streamlit dashboard

---

## 📋 Manual Setup (If Scripts Don't Work)

### Step 1: Verify Python Installation
```bash
python --version
# or
python3 --version
```
**Required:** Python 3.8 or higher

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip
```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation time:** 3-10 minutes (depending on internet speed)

### Step 5: Create Directories
```bash
mkdir -p models outputs data logs config assets
```

### Step 6: Verify Setup (Optional)
```bash
python verify_setup.py
```

This will check:
- ✅ Python version
- ✅ All dependencies installed
- ✅ GPU/CUDA availability
- ✅ Directory structure
- ✅ YOLOv8 model loading

### Step 7: Run the Application
```bash
streamlit run app.py
```

**Application will open at:** `http://localhost:8501`

---

## 🐛 Troubleshooting

### Issue: "python command not found"
**Solution:**
- Reinstall Python from https://www.python.org/
- Make sure to check "Add Python to PATH"

### Issue: Virtual environment activation fails
**Solution:**
- Delete the `venv` folder: `rmdir /s venv` (Windows) or `rm -rf venv` (Linux/Mac)
- Run the startup script again

### Issue: Dependencies installation fails
**Solution:**
```bash
# Try installing individually
pip install ultralytics
pip install streamlit
pip install opencv-python
# ... etc
```

### Issue: "No module named streamlit"
**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Streamlit not found
**Solution:**
- Make sure virtual environment is activated
- Try: `python -m streamlit run app.py`

### Issue: Model not found
**Solution:**
- Models auto-download on first use
- Or manually download: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

### Issue: GPU not detected
**Solution:**
- Check: `python verify_setup.py`
- Use CPU instead: Change `device='cuda'` to `device='cpu'` in code
- Requires: NVIDIA GPU + CUDA toolkit + cuDNN

---

## 📦 System Requirements

### Minimum (CPU Only)
- Python 3.8+
- 4 GB RAM
- 2 GB disk space

### Recommended (GPU)
- Python 3.8+
- 8 GB RAM
- 5 GB disk space
- NVIDIA GPU (GTX 1050+ or better)
- CUDA 11.8+ and cuDNN 8.6+

### Supported OS
- Windows 10+ (64-bit)
- macOS 10.14+ (Intel or Apple Silicon)
- Ubuntu 18.04+ (or other Linux distributions)

---

## ✅ Verification Checklist

After setup, verify everything works:

1. **Check Python:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Check Virtual Environment:**
   ```bash
   which python  # Should show venv path
   ```

3. **Check Dependencies:**
   ```bash
   pip list  # Should show all packages
   ```

4. **Run Verification:**
   ```bash
   python verify_setup.py
   ```

5. **Start Dashboard:**
   ```bash
   streamlit run app.py
   ```

6. **Test Dashboard:**
   - Open browser to `http://localhost:8501`
   - Load a model
   - Try uploading an image

---

## 🚀 First Time Usage

1. **Start the dashboard:**
   ```bash
   streamlit run app.py
   ```

2. **Load a model:**
   - Look at the left sidebar
   - Select "YOLOv8 Nano" (fastest)
   - Click "Load Model"
   - Wait for "Model loaded successfully!"

3. **Try image detection:**
   - Go to "Image Detection" tab
   - Click "Upload Image"
   - Select any .jpg or .png image
   - Click "Detect Objects"
   - View results and download

4. **Try other features:**
   - Video Detection tab - upload a video
   - Webcam Detection tab - use your camera
   - Analytics tab - see statistics

---

## 📚 Documentation

- **QUICKSTART.md** - Quick start guide (5 minutes)
- **README.md** - Full documentation (everything)
- **PROJECT_SUMMARY.md** - Project overview
- **This file** - Setup instructions

---

## 💡 Tips

✅ **First time slow?** Models download automatically (takes a few minutes)

✅ **Want it faster?** Use GPU: Set `device='cuda'` in config

✅ **Want smaller images?** Reduce in `config/settings.json`

✅ **Want custom models?** Place them in `models/` folder

✅ **Getting errors?** Check `logs/detection.log`

---

## 🔧 Configuration

Edit `config/settings.json` to customize:
- Confidence threshold
- Model size
- Device (CPU/GPU)
- Image dimensions
- Output paths

---

## 📞 Need Help?

1. **Check QUICKSTART.md** - Quick answers to common questions
2. **Check README.md** - Detailed documentation and troubleshooting
3. **Run verify_setup.py** - Identifies system issues
4. **Check logs/detection.log** - Debugging information

---

## 🎯 Next Steps

After successful setup:

1. Read **QUICKSTART.md** (5 minutes)
2. Try the **Streamlit dashboard** (upload an image)
3. Explore the **different detection modes**
4. Check out the **Analytics tab**
5. Read the full **README.md** for advanced features

---

## ✨ You're Ready!

Once the Streamlit dashboard opens in your browser, you're good to go! 🎉

For questions or issues, refer to the documentation files included in the project.

---

**Internship ID:** CITS2432  
**Domain:** Artificial Intelligence & Computer Vision  
**Status:** Ready to Run ✅

---

## 📝 Quick Reference

| Task | Command |
|------|---------|
| Create venv | `python -m venv venv` |
| Activate venv (Windows) | `venv\Scripts\activate` |
| Activate venv (macOS/Linux) | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Verify setup | `python verify_setup.py` |
| Start dashboard | `streamlit run app.py` |
| Access dashboard | `http://localhost:8501` |
| Test image detection | Upload in web interface |
| Stop dashboard | `Ctrl+C` in terminal |
| Deactivate venv | `deactivate` |

---

Good luck with your Real-Time Object Detection project! 🚀
