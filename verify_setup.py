"""
Installation Verification & System Check Script
Real-Time Object Detection Using YOLOv8
Internship ID: CITS2432
"""

import sys
import importlib
from pathlib import Path
import platform

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check Python version"""
    print_section("Python Version Check")
    
    version = sys.version
    major, minor, micro = sys.version_info[:3]
    
    print(f"Python Version: {version.split()[0]}")
    print(f"Version Info: {major}.{minor}.{micro}")
    
    if major >= 3 and minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python 3.8 or higher required")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def check_dependencies():
    """Check all required dependencies"""
    print_section("Dependencies Check")
    
    packages = {
        'ultralytics': 'ultralytics',
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
        'Pandas': 'pandas',
        'Streamlit': 'streamlit',
        'Matplotlib': 'matplotlib',
        'PyTorch': 'torch',
        'scikit-learn': 'sklearn',
        'PIL': 'PIL',
    }
    
    results = {}
    for pkg_name, import_name in packages.items():
        results[pkg_name] = check_package(pkg_name, import_name)
    
    installed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nInstalled: {installed_count}/{total_count}")
    return all(results.values())

def check_gpu():
    """Check GPU/CUDA availability"""
    print_section("GPU/CUDA Check")
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA Available: Yes")
            print(f"   - CUDA Version: {torch.version.cuda}")
            print(f"   - Device Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"   - Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     Memory: {props.total_memory / 1e9:.2f} GB")
            
            return True
        else:
            print(f"✅ CUDA Available: No (Will use CPU)")
            return False
    
    except Exception as e:
        print(f"⚠️ Could not check CUDA: {str(e)}")
        return False

def check_directories():
    """Check if required directories exist"""
    print_section("Directory Structure Check")
    
    required_dirs = [
        'models',
        'outputs',
        'data',
        'logs',
        'config',
        'assets'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            all_exist = False
    
    return all_exist

def check_files():
    """Check if required files exist"""
    print_section("Project Files Check")
    
    required_files = [
        'app.py',
        'utils.py',
        'detect_image.py',
        'detect_video.py',
        'webcam_detection.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md'
    ]
    
    all_exist = True
    
    for filename in required_files:
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {filename} ({size:,} bytes)")
        else:
            print(f"❌ {filename} (missing)")
            all_exist = False
    
    return all_exist

def check_yolov8_model():
    """Check if YOLOv8 model can be loaded"""
    print_section("YOLOv8 Model Check")
    
    try:
        from ultralytics import YOLO
        
        print("Attempting to load YOLOv8 Nano model...")
        print("(This may take a minute on first run)")
        
        # Try to load model
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 Model loaded successfully")
        
        # Get model info
        print(f"\nModel Information:")
        print(f"  - Model type: {model.model.__class__.__name__}")
        print(f"  - Model size: ~3.2 MB (Nano)")
        
        return True
    
    except Exception as e:
        print(f"❌ Could not load YOLOv8 model: {str(e)}")
        return False

def check_camera():
    """Check if camera/webcam is available"""
    print_section("Camera/Webcam Check")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and accessible")
            
            # Get camera properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"   - Resolution: {frame_width}x{frame_height}")
            print(f"   - FPS: {fps:.0f}")
            
            cap.release()
            return True
        else:
            print("⚠️ Camera not detected or not accessible")
            print("   (This is okay if you don't plan to use webcam)")
            return False
    
    except Exception as e:
        print(f"⚠️ Could not check camera: {str(e)}")
        return False

def check_streamlit():
    """Check if Streamlit works"""
    print_section("Streamlit Check")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit imported successfully")
        
        # Check if we can import streamlit components
        from streamlit import session_state
        print(f"✅ Streamlit components available")
        
        return True
    
    except Exception as e:
        print(f"❌ Streamlit error: {str(e)}")
        return False

def generate_report(checks):
    """Generate final report"""
    print_section("System Compatibility Report")
    
    total = len(checks)
    passed = sum(checks.values())
    failed = total - passed
    
    print(f"\nTotal Checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    success_rate = (passed / total) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    print_section("Recommendations")
    
    if success_rate == 100:
        print("✅ All checks passed! Your system is ready.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Load a model and start detecting objects!")
    
    elif success_rate >= 80:
        print("⚠️ Most checks passed, but some issues detected.")
        print("\nYour system should work but may have limitations:")
        
        if not checks.get('gpu', False):
            print("  - GPU not available (will use CPU, slower)")
        
        if not checks.get('camera', False):
            print("  - Camera not detected (can't use webcam)")
        
        print("\nYou can still:")
        print("  1. Run: streamlit run app.py")
        print("  2. Use image and video detection")
    
    else:
        print("❌ Critical issues detected.")
        print("\nPlease fix the following:")
        print("  - Install Python 3.8 or higher")
        print("  - Install required dependencies: pip install -r requirements.txt")
        print("  - Create required directories")
        print("\nThen run this script again.")
    
    return success_rate >= 80

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("  Real-Time Object Detection System - Verification Script")
    print("  Internship ID: CITS2432")
    print("="*60)
    
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Python Location: {sys.executable}")
    
    checks = {}
    
    # Run all checks
    checks['python'] = check_python_version()
    checks['dependencies'] = check_dependencies()
    checks['gpu'] = check_gpu()
    checks['directories'] = check_directories()
    checks['files'] = check_files()
    checks['streamlit'] = check_streamlit()
    checks['camera'] = check_camera()
    
    # Try to load YOLOv8 model
    try:
        checks['yolo_model'] = check_yolov8_model()
    except:
        checks['yolo_model'] = False
        print("\n⚠️ Note: Model will be downloaded on first use if not available")
    
    # Generate final report
    ready = generate_report(checks)
    
    # Final message
    print_section("Final Status")
    if ready:
        print("✅ Your system is ready to run the object detection project!")
    else:
        print("❌ Please fix the issues above before running the project.")
    
    print("\n" + "="*60 + "\n")
    
    return 0 if ready else 1

if __name__ == "__main__":
    sys.exit(main())
