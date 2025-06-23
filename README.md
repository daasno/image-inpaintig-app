# Image Inpainting Application - Professional Edition with Batch Processing

A professional image restoration and object removal tool with advanced features, modern UI, and powerful batch processing capabilities.

![Application Screenshot](logo.png)

## 🌟 Features

### Core Functionality

- **Advanced Image Inpainting**: Remove unwanted objects or restore damaged areas in images
- **Dual Processing Modes**: Single image and batch processing support
- **Dual Implementation**: CPU and GPU (CUDA) acceleration support
- **Multiple Algorithms**: Optimized patch-based inpainting with configurable parameters
- **High-Quality Results**: Professional-grade image restoration capabilities
- **Batch Processing**: Process multiple image-mask pairs automatically with parallel processing

### 🎨 User Interface & Experience

- **Modern Dark Theme**: Eye-friendly dark interface with excellent image visibility
- **Dual Mode Interface**: Tabbed interface for single image and batch processing modes
- **Interactive Mask Editor**: Built-in drawing tools for creating custom masks
- **Tabbed Image Viewer**: Switch between input, mask, and result images
- **Real-time Preview**: See mask overlays while drawing
- **Responsive Design**: Adaptive layout that works on different screen sizes
- **Batch Management Panel**: Comprehensive controls for folder selection and batch operations

### 🛠️ Advanced Tools

#### Single Image Mode
- **Built-in Mask Creator**: Draw masks directly on your images with brush and eraser tools
- **Enhanced Mask Preview**: See masks with clear red/gray visualization before processing
- **Mask Confirmation Dialog**: Review and approve masks before inpainting
- **Exhaustive Research Mode**: Automatically find optimal parameter combinations
- **Processing Time Analytics**: Bar chart visualization of performance metrics
- **Adjustable Brush Sizes**: Precise control with sizes from 5-100 pixels
- **Opacity Control**: Adjust mask overlay visibility for better precision

#### Batch Processing Mode
- **Automatic File Matching**: Smart pairing of image and mask files based on naming patterns
- **Folder-based Organization**: Select separate folders for images, masks, and results
- **Real-time Progress Tracking**: Monitor batch processing with detailed progress indicators
- **Parallel Processing**: Efficient multi-threaded batch operations
- **Error Handling**: Comprehensive validation and error reporting for each pair
- **Batch Statistics**: Detailed summary of processing results and performance
- **Flexible File Naming**: Supports various naming conventions (img1.jpg, mask1.png, etc.)

### 🔧 Technical Features

- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **Memory Management**: Intelligent handling of large images with safety limits
- **Enhanced Error Handling**: Robust error recovery and user-friendly error messages
- **Settings Persistence**: Remembers your preferences and recent files
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Thread-Safe Processing**: Safe concurrent operations for batch processing
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📋 Requirements

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- For GPU acceleration: NVIDIA GPU with CUDA support
- Storage space for input, mask, and result images (batch processing)

### Dependencies
```
PySide6>=6.0.0
opencv-python>=4.5.0
numpy>=1.20.0
numba>=0.56.0
numba[cuda]  # Optional, for GPU acceleration
matplotlib>=3.5.0  # For research analytics
seaborn>=0.11.0    # For enhanced visualization
pandas>=1.3.0      # For data handling
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-inpainting-app
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **For GPU support** (optional):
   ```bash
   pip install numba[cuda]
   ```

## 🎯 Quick Start

### Single Image Mode

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Load an image**:
   - Click "📁 Load Image" or press `Ctrl+O`
   - Select your image file (PNG, JPG, BMP, TIFF supported)

3. **Create or load a mask**:
   - **Option A**: Click "✏️ Create Mask" to draw directly on your image with built-in editor
   - **Option B**: Click "🎭 Load Mask" to use an existing mask file
   - **Option C**: Use "🔬 Exhaustive Research" to automatically find optimal settings

4. **Adjust parameters** (optional):
   - **Patch Size**: Controls texture analysis detail (3-21, default: 9)
   - **P-Value**: Distance metric for matching (0.1-10.0, default: 1.0)
   - **Implementation**: Choose CPU or GPU acceleration

5. **Run inpainting**:
   - Click "▶️ Run Inpainting" or press `F5`
   - Wait for processing to complete

6. **Save result**:
   - Click "💾 Save" or press `Ctrl+S`
   - Choose your output location and format

### Batch Processing Mode

1. **Launch the batch-enabled application**:
   ```bash
   python main_batch.py
   ```

2. **Switch to Batch Processing tab** or press `Ctrl+2`

3. **Set up folder structure**:
   - **Images Folder**: Contains your input images (img1.jpg, img2.png, etc.)
   - **Masks Folder**: Contains corresponding masks (mask1.jpg, mask2.png, etc.)
   - **Results Folder**: Where processed images will be saved

4. **File naming convention**:
   - Images: `img1.jpg`, `img2.png`, `img15.jpeg`, etc.
   - Masks: `mask1.jpg`, `mask2.png`, `mask15.jpeg`, etc.
   - Results: `result1.jpg`, `result2.jpg`, etc. (automatically generated)

5. **Configure batch settings**:
   - **Patch Size**: Apply to all images in batch
   - **P-Value**: Distance metric for all processing
   - **Implementation**: CPU or GPU for batch processing

6. **Start batch processing**:
   - Click "▶️ Start Batch Processing" or press `Ctrl+B`
   - Monitor progress in real-time
   - View detailed statistics upon completion

## 🎨 Using the Mask Editor

The built-in mask editor allows you to create precise masks directly on your images:

### Tools Available

- **🖌️ Brush Tool**: Draw white areas (regions to inpaint)
- **🧽 Eraser Tool**: Remove mask areas (make them black)
- **Size Slider**: Adjust brush/eraser size (5-100 pixels)
- **Opacity Slider**: Control mask overlay visibility (10-90%)
- **🗑️ Clear All**: Remove entire mask and start over

### Usage Tips

- **White areas in mask = regions to inpaint, black areas = preserved regions**
- **Red overlay** on image = areas that will be inpainted
- **Gray areas** = preserved regions (clear visualization)
- Use smaller brush sizes for precise work around edges
- Use larger brush sizes for filling large areas quickly
- Adjust opacity to see the underlying image clearly
- **Confirmation dialog** appears after mask creation for review
- Click "Yes" if satisfied with mask, "No" to recreate it

### Keyboard Shortcuts in Mask Editor

- `Left Click + Drag`: Draw/Erase mask
- `Mouse Wheel`: Adjust brush size
- `B`: Switch to Brush tool
- `E`: Switch to Eraser tool
- `Ctrl+A`: Clear all mask
- `Enter`: Apply mask
- `Escape`: Cancel mask editor

## 📁 Batch Processing Guide

### Setting Up Batch Processing

1. **Organize your files** in separate folders:
   ```
   project/
   ├── images/          # Input images
   │   ├── img1.jpg
   │   ├── img2.png
   │   └── img3.jpeg
   ├── masks/           # Corresponding masks
   │   ├── mask1.jpg
   │   ├── mask2.png
   │   └── mask3.jpeg
   └── results/         # Output folder (can be empty)
   ```

2. **File naming requirements**:
   - Images must start with "img" followed by a number
   - Masks must start with "mask" followed by the same number
   - Extensions can be mixed (.jpg, .png, .jpeg, .bmp, .tiff)
   - Examples: `img1.jpg` ↔ `mask1.png`, `img25.jpeg` ↔ `mask25.jpg`

3. **Batch processing features**:
   - **Automatic matching**: Files are paired by number automatically
   - **Validation**: Each pair is validated before processing
   - **Progress tracking**: Real-time progress with current pair information
   - **Error handling**: Failed pairs are logged, processing continues
   - **Summary report**: Detailed results at completion

### Batch Processing Tips

- **Use consistent naming**: Follow the img{N}/mask{N} pattern
- **Check file formats**: Ensure all images are in supported formats
- **Verify dimensions**: Each image and its corresponding mask must have matching dimensions
- **Monitor progress**: Use the progress panel to track processing status
- **Review results**: Check the summary report for any failed pairs
- **GPU acceleration**: Use GPU implementation for faster batch processing

## 🔬 Exhaustive Research Mode

The application includes an advanced research mode that automatically finds optimal parameter combinations:

### How It Works

1. **Click "🔬 Exhaustive Research"** after loading image and mask
2. **Configure test parameters**:
   - Patch sizes to test (e.g., 5, 7, 9, 11, 13)
   - P-values to test (e.g., 1.0, 1.5, 2.0, 2.5)
   - Implementation types (CPU, GPU, or both)
3. **Automatic processing**: Tests all combinations systematically
4. **Performance visualization**: Bar chart showing processing times
5. **Results analysis**: Compare quality and speed of different settings

### Features

- **Batch Processing**: Tests multiple parameter combinations automatically
- **Time Tracking**: Measures processing time for each combination
- **Visual Analytics**: Bar chart with GPU (blue) vs CPU (red) performance
- **Export Options**: Save charts as PNG, PDF, or SVG formats
- **Best Settings Identification**: Helps find optimal parameters for your specific use case

### Usage Tips

- Start with fewer combinations for initial testing
- GPU implementation typically 10-20x faster than CPU
- Smaller patch sizes process faster but may produce different quality
- Use this mode to find the sweet spot between quality and speed

## ⌨️ Keyboard Shortcuts

### File Operations

- `Ctrl+O`: Load Input Image
- `Ctrl+M`: Load Mask Image
- `Ctrl+S`: Save Result Image
- `Ctrl+Q`: Exit Application

### Processing

- `F5`: Run Inpainting
- `Ctrl+B`: Start Batch Processing
- `Ctrl+R`: Reset All

### View Controls

- `Ctrl+1`: Single Image Mode
- `Ctrl+2`: Batch Processing Mode
- `Ctrl+=`: Zoom In
- `Ctrl+-`: Zoom Out
- `Ctrl+0`: Zoom to Fit

### Help

- `F1`: Show Help Dialog

## 🔧 Parameter Guide

### Patch Size (3-21)

Controls the size of texture patches used for inpainting analysis:

- **Smaller values (3-7)**: Good for fine details and small objects. Faster processing.
- **Medium values (9-13)**: Balanced approach suitable for most images.
- **Larger values (15-21)**: Better for large areas and smooth textures. Slower processing.

### Minkowski Order (P-Value: 1.0-10.0)

Controls the distance metric used for patch matching:

- **p = 1.0 (Manhattan distance)**: More conservative, preserves structure better
- **p = 2.0 (Euclidean distance)**: Standard distance metric, balanced approach

### Implementation Choice

- **CPU Implementation**: Compatible with all systems, slower processing
- **GPU Implementation**: Requires NVIDIA GPU with CUDA, 10-20x faster

## 🏗️ Architecture

The application follows a clean modular architecture with batch processing support:

```
├── main.py                    # Single image mode entry point
├── main_batch.py              # Batch processing mode entry point
├── config/                    # Configuration and settings
│   ├── settings.py
│   ├── logging_config.py
│   └── memory_manager.py
├── models/                    # Data models and business logic
│   ├── image_data.py          # Single image data model
│   ├── batch_data.py          # Batch processing data model
│   ├── inpaint_worker.py      # Single image worker
│   └── batch_worker.py        # Batch processing worker
├── views/                     # UI components
│   ├── main_window.py         # Single image main window
│   ├── main_window_batch.py   # Batch-enabled main window
│   └── widgets/
│       ├── control_panel.py   # Processing controls
│       ├── batch_panel.py     # Batch processing controls
│       ├── image_label.py     # Image display widget
│       ├── mask_editor.py     # Mask creation tool
│       └── help_dialog.py     # Help system
└── controllers/               # Application logic coordination
    ├── app_controller.py      # Single image controller
    └── batch_controller.py    # Batch processing controller
```

For detailed architecture information, see [README_ARCHITECTURE.md](README_ARCHITECTURE.md).

## 🎨 Dark Theme

The application features a modern dark theme designed for:

- **Better Image Visibility**: Dark backgrounds provide excellent contrast for viewing images
- **Eye Comfort**: Reduced eye strain during extended use
- **Professional Appearance**: Modern, sleek interface design
- **Consistent Styling**: Unified dark theme across all components

## 🆘 Troubleshooting

### Common Issues

**GPU Not Available**

- Ensure you have an NVIDIA GPU with CUDA support
- Install CUDA toolkit and drivers
- Install numba with CUDA support: `pip install numba[cuda]`
- Use CPU implementation as fallback

**Poor Inpainting Results**

- Try different patch sizes (start with 9, then try 7 or 13)
- Adjust p-value (try 1.0 for Manhattan distance)
- Ensure mask doesn't cover too large an area
- Check that surrounding areas have sufficient texture/detail

**Slow Processing**

- Use GPU implementation if available
- Reduce patch size (try 7 or 5)
- Resize image to smaller dimensions before processing
- Reduce the size of masked areas

**Memory Issues**

- Reduce image size before processing
- Use smaller patch sizes
- Close other applications to free memory
- Switch from GPU to CPU implementation

**Batch Processing Issues**

- **Files not matching**: Check file naming convention (img{N}.ext ↔ mask{N}.ext)
- **Dimension mismatch**: Ensure image and mask have same dimensions
- **Permission errors**: Verify write access to results folder
- **Large batch processing**: Use GPU implementation and ensure sufficient RAM
- **Mixed file formats**: Supported but ensure all files are valid images

For more detailed troubleshooting, press `F1` in the application to access the comprehensive help system.

## 📝 Recent Updates

### Version 1.3.0 - Batch Processing Edition

#### 🆕 Major New Features

- **Batch Processing Mode**: Complete batch processing system for multiple image pairs
- **Dual Mode Interface**: Tabbed interface supporting both single and batch operations
- **Automatic File Matching**: Smart pairing based on filename patterns (img{N} ↔ mask{N})
- **Parallel Processing**: Multi-threaded batch operations with progress tracking
- **Comprehensive Batch Management**: Folder selection, validation, and result organization
- **New Entry Points**: Separate `main_batch.py` for batch-focused workflows

#### 🎯 Batch Processing Features

- **Folder-based Organization**: Separate input folders for images, masks, and results
- **Real-time Progress**: Live progress tracking with current pair information
- **Error Handling**: Individual pair validation with detailed error reporting
- **Batch Statistics**: Comprehensive summary reports upon completion
- **Flexible Naming**: Support for various file extensions and numbering schemes
- **Memory Efficient**: Optimized processing for large batch operations

#### 🏗️ Architecture Enhancements

- **New Models**: `BatchData` class for batch operation management
- **New Workers**: `BatchInpaintWorker` for threaded batch processing
- **New Controllers**: `BatchAppController` for enhanced application logic
- **New UI Components**: `BatchPanel` widget for batch operation controls
- **Enhanced Main Window**: `BatchEnabledMainWindow` with dual-mode support

#### 🔧 Technical Improvements

- **Modular Design**: Clean separation between single and batch processing logic
- **Thread Safety**: Safe concurrent operations for batch processing
- **Enhanced Error Handling**: Robust validation and error recovery for batch operations
- **Improved Logging**: Detailed logging for batch processing monitoring
- **Memory Management**: Optimized memory usage for large batch operations

### Version 1.2.0 - Professional Edition

#### 🎨 User Interface Improvements

- **Enhanced Mask Preview System**: Red/gray visualization for clear mask identification
- **Tabbed Image Interface**: Input, Mask, and Result tabs with smart enable/disable
- **Modern Welcome Dialog**: Streamlined dark theme onboarding experience
- **Mask Confirmation Dialog**: Review masks before processing with clear visualization
- **Responsive Design**: Welcome dialog adapts to different screen sizes

#### 🛠️ New Features

- **Exhaustive Research Mode**: Automatically test multiple parameter combinations
- **Performance Analytics**: Interactive bar charts showing processing times
- **Chart Export Options**: Save performance data as PNG, PDF, or SVG
- **Enhanced Mask Creation**: Built-in editor with confirmation workflow
- **Real-time Mask Preview**: See exactly what will be inpainted vs preserved

#### 🏗️ Technical Improvements

- **Advanced Plotting Integration**: Matplotlib and Seaborn for data visualization
- **Improved Error Handling**: Fixed QTextCursor compatibility issues
- **Memory Management**: Better handling of large images and batch processing
- **Progress Dialog Enhancement**: Cleaner status reporting during processing

#### 🔧 Quality of Life

- **Automatic Tab Switching**: Smart navigation to relevant image views
- **Welcome Dialog Control**: Option to disable welcome screen permanently
- **Parameter Research**: Find optimal settings through systematic testing
- **Visual Feedback**: Clear indication of mask vs preserved areas
- **Professional Workflow**: Complete mask creation and approval process

### Version 1.1.0 - Enhanced Edition

#### Legacy Features

- **Dark Theme**: Complete dark theme implementation for better image visibility
- **Interactive Mask Editor**: Built-in drawing tools for creating custom masks
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Cross-Platform Compatibility**: Enhanced support for different operating systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PySide6 for the modern Qt-based interface
- OpenCV for image processing capabilities
- Numba for high-performance computing acceleration
- CUDA for GPU acceleration support

---

**Note**: This application is designed for educational and research purposes. For commercial use, please ensure compliance with all relevant licenses and regulations. 