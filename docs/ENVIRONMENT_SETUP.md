# 🚀 Environment Setup Guide for Skin Type Classification

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (Automated)](#quick-setup-automated)
3. [Manual Setup](#manual-setup)
4. [Package Dependencies](#package-dependencies)
5. [Kernel Selection](#kernel-selection)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Quick Start Checklist](#quick-start-checklist)

## 📋 Prerequisites

### **System Requirements:**
- **Python**: 3.8 or higher (3.13.3 recommended)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: At least 8GB RAM (16GB+ recommended for training)
- **Storage**: 5GB free space for dataset and models
- **GPU**: Optional but recommended (CUDA-compatible for faster training)

### **Required Software:**
- Python 3.8+
- pip (Python package manager)
- Git (for cloning repositories)
- VS Code with Python extension (recommended)

## 🚀 Quick Setup (Automated)

If you're using VS Code with the Python extension, the environment has been pre-configured:

```bash
# The virtual environment is already created at:
# /Users/bjit/Desktop/My_Files/Projects/Image_Analyzer/kaggle_image_analyzer/.venv/

# Kernel registered as: "Python 3.13 (Skin Classification)"
```

**Just select the correct kernel in VS Code and start coding!**

## 🛠️ Manual Setup

### **Step 1: Clone/Download Project**
```bash
# If using Git:
git clone <repository-url>
cd kaggle_image_analyzer

# Or download and extract the project files
```

### **Step 2: Create Virtual Environment**
```bash
# Navigate to project directory
cd /path/to/kaggle_image_analyzer

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### **Step 3: Upgrade pip**
```bash
pip install --upgrade pip
```

### **Step 4: Install PyTorch**

#### **For CPU-only systems:**
```bash
pip install torch torchvision
```

#### **For GPU systems with CUDA:**
```bash
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Check CUDA availability after installation:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Step 5: Install Other Dependencies**
```bash
# Install all required packages
pip install numpy pandas scikit-learn matplotlib seaborn pillow jupyter ipykernel

# Or install from requirements file (if available):
pip install -r requirements.txt
```

### **Step 6: Register Jupyter Kernel**
```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name skin-classification --display-name "Python 3.13 (Skin Classification)"

# Verify kernel installation:
jupyter kernelspec list
```

### **Step 7: Verify Installation**
```bash
python -c "
import torch, torchvision, numpy, pandas, sklearn, matplotlib, seaborn
from PIL import Image
print('✅ All packages installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## 📦 Package Dependencies

### **Core Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.7.1+ | Deep learning framework |
| `torchvision` | 0.22.1+ | Computer vision utilities |
| `numpy` | 2.3.1+ | Numerical computing |
| `pandas` | 2.3.1+ | Data manipulation and analysis |
| `scikit-learn` | 1.7.0+ | Machine learning metrics and tools |
| `matplotlib` | 3.10.3+ | Plotting and visualization |
| `seaborn` | 0.13.2+ | Statistical data visualization |
| `pillow` | 11.3.0+ | Image processing library |

### **Jupyter Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `ipykernel` | 6.29.5+ | Jupyter kernel support |
| `jupyter` | 1.1.1+ | Jupyter notebook environment |

### **Optional Dependencies:**
- `tqdm`: Progress bars for training loops
- `tensorboard`: Training visualization
- `opencv-python`: Advanced image processing

## 🔧 Kernel Selection

### **In VS Code:**
1. Open the notebook file (`skin-type-classification.ipynb`)
2. Click on the kernel selector in the top-right corner
3. Select **"Python 3.13 (Skin Classification)"** from the dropdown
4. If not available, select any Python 3.8+ kernel with required packages

### **In Jupyter Lab/Notebook:**
1. Open the notebook
2. Go to `Kernel` → `Change Kernel`
3. Select **"Python 3.13 (Skin Classification)"**

### **Alternative Kernels:**
If the custom kernel isn't available, look for:
- Python 3.13.x (recommended)
- Python 3.12.x
- Python 3.11.x
- Python 3.10.x
- Python 3.9.x
- Python 3.8.x (minimum)

**⚠️ Avoid:** "Python 3.12.11" if it shows ipykernel errors

## 📁 Project Structure

Your project should have the following structure:

```
kaggle_image_analyzer/
├── README.md                         # Project documentation
├── ENVIRONMENT_SETUP.md             # This file
├── skin-type-classification.ipynb    # Main notebook
├── requirements.txt                  # Package dependencies (optional)
├── .venv/                           # Virtual environment
│   ├── bin/                         # Scripts (macOS/Linux)
│   ├── Scripts/                     # Scripts (Windows)
│   ├── lib/                         # Installed packages
│   └── ...
├── trained_models/                   # Pre-trained model files
│   ├── best_skin_model.pth          # Model weights only
│   ├── best_skin_model_entire.pth   # Complete model
│   ├── label_maps.pkl               # Label mappings
│   └── training_stats.pkl           # Training statistics
└── Oily-Dry-Skin-Types/            # Dataset folder
    ├── README.dataset.txt           # Dataset information
    ├── train/                       # Training images
    │   ├── dry/                     # Dry skin images
    │   ├── normal/                  # Normal skin images
    │   └── oily/                    # Oily skin images
    ├── valid/                       # Validation images
    │   ├── dry/
    │   ├── normal/
    │   └── oily/
    └── test/                        # Test images
        ├── dry/
        ├── normal/
        └── oily/
```

## 🔍 Troubleshooting

### **Common Issues and Solutions:**

#### **Issue 1: "No module named 'torch'"**
```bash
# Solution: Install PyTorch
pip install torch torchvision

# For specific CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Issue 2: "Kernel died" or ipykernel errors**
```bash
# Solution 1: Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Solution 2: Re-register kernel
python -m ipykernel install --user --name skin-classification --display-name "Python 3.13 (Skin Classification)" --force-reinstall
```

#### **Issue 3: "CUDA out of memory"**
```python
# Solution: Force CPU usage in notebook
device = torch.device("cpu")

# Or reduce batch size
batch_size = 16  # Instead of 32
```

#### **Issue 4: "Permission denied" when creating virtual environment**
```bash
# Solution: Use --user flag or different location
python3 -m venv ~/.venvs/skin-classification
source ~/.venvs/skin-classification/bin/activate
```

#### **Issue 5: Missing dataset files**
- Download the skin type dataset from Kaggle
- Extract to `Oily-Dry-Skin-Types/` folder
- Ensure folder structure matches the expected layout

#### **Issue 6: Import errors for specific packages**
```bash
# Solution: Install missing packages individually
pip install matplotlib seaborn scikit-learn pandas numpy pillow

# Check for conflicts:
pip check
```

### **Environment Conflicts:**
```bash
# Reset environment completely:
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# ... reinstall packages
```

## ⚡ Performance Optimization

### **For CPU-only Systems:**
```python
# Reduce computational load:
batch_size = 16        # Instead of 32
num_epochs = 10        # Instead of 30
num_workers = 2        # For DataLoader
```

### **For GPU Systems:**
```python
# Verify GPU usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Monitor GPU memory:
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### **Memory Management:**
```python
# Clear cache periodically:
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use gradient checkpointing for large models:
model.gradient_checkpointing_enable()  # If supported
```

## ✅ Quick Start Checklist

Before running the notebook, ensure:

### **Environment Setup:**
- [ ] ✅ Python 3.8+ installed and accessible
- [ ] ✅ Virtual environment created and activated
- [ ] ✅ All required packages installed without errors
- [ ] ✅ PyTorch installed with appropriate CUDA support (if available)

### **Jupyter Setup:**
- [ ] ✅ ipykernel installed in virtual environment
- [ ] ✅ Jupyter kernel registered and selectable
- [ ] ✅ Correct kernel selected in VS Code/Jupyter

### **Project Files:**
- [ ] ✅ Notebook file (`skin-type-classification.ipynb`) accessible
- [ ] ✅ Dataset folder (`Oily-Dry-Skin-Types/`) with images
- [ ] ✅ Trained models folder (`trained_models/`) with .pth files
- [ ] ✅ Sufficient disk space (5GB+ free)

### **System Resources:**
- [ ] ✅ At least 8GB RAM available
- [ ] ✅ Stable internet connection (for downloading models)
- [ ] ✅ No resource-intensive applications running

## 🚀 Ready to Start!

Once you've completed the checklist:

### **Step 1: Test Environment**
Run the verification cell in the notebook to ensure everything works:

```python
# This cell is included in the notebook
import torch, numpy, pandas, sklearn, matplotlib, seaborn
from PIL import Image
print("✅ Environment ready!")
```

### **Step 2: Start the Notebook**
1. Open `skin-type-classification.ipynb` in VS Code
2. Select the "Python 3.13 (Skin Classification)" kernel
3. Run cells sequentially starting from the imports
4. Monitor outputs for any errors

### **Step 3: Monitor Performance**
- Watch memory usage during training
- Check GPU utilization (if using GPU)
- Save your work frequently (Ctrl+S)

## 📞 Support

If you encounter issues not covered in this guide:

1. **Check VS Code Output**: View → Output → Python/Jupyter
2. **Verify Package Versions**: Run the verification cell
3. **Check System Resources**: Monitor RAM/CPU usage
4. **Review Error Messages**: Look for specific import/runtime errors

## 🎯 Success!

Your environment is now ready for skin type classification! 

**Happy coding! 🚀**

---

*Last updated: January 2025*
