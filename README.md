# Image Inpainting Application - Professional Edition with Advanced Research & Comparison

A professional image restoration and object removal tool with advanced features, modern UI, powerful batch processing capabilities, and comprehensive quality analysis tools.

![Application Screenshot](images/logo.png)

## 🌟 Features

### Core Functionality

- **Advanced Image Inpainting**: Remove unwanted objects or restore damaged areas in images
- **Triple Processing Modes**: Single image, batch processing, and comparison analysis support
- **Dual Implementation**: CPU and GPU (CUDA) acceleration support
- **Multiple Algorithms**: Optimized patch-based inpainting with configurable parameters
- **High-Quality Results**: Professional-grade image restoration capabilities
- **Batch Processing**: Process multiple image-mask pairs automatically with parallel processing
- **Quality Analysis**: Comprehensive metrics including PSNR, SSIM, LPIPS, and FID

### 🎨 User Interface & Experience

- **Modern Dark Theme**: Eye-friendly dark interface with excellent image visibility
- **Triple Mode Interface**: Tabbed interface for single image, batch processing, and comparison modes
- **Interactive Mask Editor**: Built-in drawing tools for creating custom masks
- **Tabbed Image Viewer**: Switch between input, mask, and result images
- **Real-time Preview**: See mask overlays while drawing
- **Responsive Design**: Adaptive layout that works on different screen sizes
- **Batch Management Panel**: Comprehensive controls for folder selection and batch operations
- **Comparison Panel**: Side-by-side image comparison with quality metrics visualization

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
- **Batch Exhaustive Research**: Run parameter optimization across multiple image pairs
- **Quality Metrics**: Calculate PSNR, SSIM, and LPIPS for all processed images in batch mode
- **FID Calculation**: Compute FID scores for large batches (50+ images) to assess distribution quality

#### Comparison Mode (NEW!)
- **Side-by-Side Comparison**: Visual comparison of original and inpainted images
- **Quality Metrics Calculation**: PSNR, SSIM, and LPIPS calculation with quality interpretation
- **SSIM Difference Visualization**: Heat map showing structural differences
- **Quality Interpretation**: Automatic quality assessment (Excellent/Good/Fair/Poor)
- **Comparison Reports**: Export detailed analysis reports
- **Perceptual Metrics**: Advanced LPIPS metric for human-aligned quality assessment
- **Metrics History**: Track quality improvements across different parameter settings

### 🔧 Technical Features

- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **Memory Management**: Intelligent handling of large images with safety limits
- **Enhanced Error Handling**: Robust error recovery and user-friendly error messages
- **Settings Persistence**: Remembers your preferences and recent files
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Thread-Safe Processing**: Safe concurrent operations for batch processing
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Quality Metrics Integration**: Built-in PSNR, SSIM, LPIPS, and FID calculation
  - Traditional metrics (PSNR, SSIM) via scikit-image
  - Perceptual metrics (LPIPS) via deep learning
  - Distribution metrics (FID) for batch evaluation (50+ images)

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
scikit-image>=0.19.0  # For quality metrics (PSNR, SSIM)

# Advanced Perceptual Metrics (Optional but Recommended)
torch>=1.10.0      # For LPIPS and FID
torchvision>=0.11.0  # For LPIPS and FID
lpips>=0.1.4       # For perceptual similarity metric
```

**Note**: LPIPS and FID are optional but highly recommended for advanced quality assessment. See [METRICS_GUIDE.md](METRICS_GUIDE.md) for details.

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

### Comparison Mode (NEW!)

1. **Launch the batch-enabled application**:
   ```bash
   python main_batch.py
   ```

2. **Switch to Comparison Mode tab** or press `Ctrl+3`

3. **Load images for comparison**:
   - **Original Image**: Click "📁 Load Original" to load the reference image
   - **Inpainted Image**: Click "📁 Load Inpainted" to load the processed image

4. **Calculate quality metrics**:
   - Click "📊 Calculate Metrics" to compute PSNR and SSIM values
   - View side-by-side comparison and SSIM difference visualization
   - Get automatic quality interpretation

5. **Export comparison results**:
   - Click "💾 Save Comparison" to export detailed analysis report
   - Report includes metrics, quality assessment, and image information

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

### Single Image Research

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

### Batch Exhaustive Research (NEW!)

The application now supports running exhaustive parameter research across multiple image pairs:

#### How It Works

1. **Access via Batch Processing Mode**: Switch to batch mode and select image pairs
2. **Configure Research Parameters**:
   - Select patch sizes to test (e.g., 5, 7, 9, 11, 13)
   - Choose P-values to test (e.g., 1.0, 1.5, 2.0, 2.5)
   - Select implementation types (CPU, GPU, or both)
   - Choose research strategy (all pairs, first N pairs, or selected pairs)
3. **Advanced Options**:
   - Enable quality metrics calculation (PSNR/SSIM)
   - Set best result criteria (fastest, best PSNR, best SSIM)
   - Configure export options (CSV results, best images)
4. **Automated Processing**: Tests all combinations across selected image pairs
5. **Comprehensive Results**: Detailed analysis with quality metrics and timing data

#### Features

- **Flexible Pair Selection**: Choose all pairs, first N pairs, or manually select specific pairs
- **Quality Metrics Integration**: Optional PSNR/SSIM calculation for each result
- **Multiple Optimization Criteria**: Find best results based on speed or quality
- **Comprehensive Reporting**: Export results to CSV with detailed metrics
- **Best Image Export**: Automatically save the best inpainting results
- **Progress Monitoring**: Real-time progress tracking with detailed status updates
- **Error Resilience**: Continue processing even if some combinations fail

#### Usage Tips

- Start with a small subset of pairs for initial parameter exploration
- Enable metrics calculation for quality-focused research (slower but more informative)
- Use "First N Pairs" strategy for quick parameter validation
- Export CSV results for further analysis in spreadsheet applications
- GPU implementation typically provides 10-20x speed improvement for batch research

## 📊 Quality Metrics & Comparison

### Supported Metrics

The application calculates several image quality metrics with different purposes:

#### Traditional Metrics (Always Available)
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level similarity
  - Higher values indicate better quality
  - Excellent: ≥40 dB, Good: 30-40 dB, Fair: 20-30 dB, Poor: <20 dB
  - Fast computation, pixel-based
  
- **SSIM (Structural Similarity Index)**: Measures structural similarity
  - Values range from 0 to 1, higher is better
  - Excellent: ≥0.95, Good: 0.85-0.95, Fair: 0.7-0.85, Poor: <0.7
  - Considers luminance, contrast, and structure
  - Includes difference map visualization
  
- **MSE (Mean Squared Error)**: Basic pixel difference measurement
  - Lower is better
  - Simple and fast computation

#### Advanced Perceptual Metrics (Requires PyTorch) ✨ NEW
- **LPIPS (Learned Perceptual Image Patch Similarity)**
  - Deep learning-based perceptual metric
  - Values typically 0-1, **LOWER is BETTER**
  - Excellent: ≤0.1, Good: ≤0.3, Fair: ≤0.5, Poor: >0.5
  - More aligned with human perception than PSNR/SSIM
  - GPU accelerated when available
  - **Use for**: Single image quality assessment, perceptual similarity

- **FID (Fréchet Inception Distance)**
  - Distribution-based metric for image sets
  - Lower is better (measures distribution distance)
  - **Requires 50+ images** for reliable results
  - Uses Inception v3 features
  - **Use for**: Batch quality assessment, parameter optimization
  - **Limitation**: Cannot be computed on small image sets (<50 images)

**📘 For detailed information**, see [METRICS_GUIDE.md](METRICS_GUIDE.md)

### Comparison Features

- **Visual Comparison**: Side-by-side image display with synchronized zooming
- **SSIM Difference Map**: Heat map visualization showing structural differences
- **Quality Interpretation**: Automatic assessment of inpainting quality
- **Detailed Reports**: Comprehensive analysis including image properties and metrics
- **Export Capabilities**: Save comparison results and visualizations
- **LPIPS Integration**: Optional perceptual metric calculation (checkbox in UI)

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
- `Ctrl+3`: Comparison Mode
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

The application follows a clean modular architecture with batch processing and comparison support:

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
│   ├── comparison_data.py     # Comparison data model (NEW!)
│   ├── metrics.py             # Quality metrics calculation (NEW!)
│   ├── inpaint_worker.py      # Single image worker
│   ├── batch_worker.py        # Batch processing worker
│   └── batch_exhaustive_worker.py  # Batch exhaustive research worker (NEW!)
├── views/                     # UI components
│   ├── main_window.py         # Single image main window
│   ├── main_window_batch.py   # Batch-enabled main window
│   ├── dialogs/
│   │   ├── exhaustive_research_dialog.py     # Single image research
│   │   └── batch_exhaustive_dialog.py        # Batch research (NEW!)
│   └── widgets/
│       ├── control_panel.py   # Processing controls
│       ├── batch_panel.py     # Batch processing controls
│       ├── comparison_panel.py # Image comparison widget (NEW!)
│       ├── image_label.py     # Image display widget
│       ├── mask_editor.py     # Mask creation tool
│       └── help_dialog.py     # Help system
└── controllers/               # Application logic coordination
    ├── app_controller.py      # Single image controller
    ├── batch_controller.py    # Batch processing controller
    └── comparison_controller.py # Comparison controller (NEW!)
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

**Comparison Mode Issues**

- **Images not displaying**: Ensure both images are in supported formats (PNG, JPG, TIFF, etc.)
- **Metrics calculation fails**: Verify images have the same dimensions
- **Poor quality scores**: Check if images are properly aligned and from the same source
- **Export errors**: Ensure you have write permissions to the selected directory

**Batch Exhaustive Research Issues**

- **Long processing times**: Use GPU implementation and reduce parameter combinations
- **Memory issues**: Process smaller batches or reduce image sizes
- **Failed combinations**: Check individual error messages in the progress log
- **Export failures**: Verify sufficient disk space and write permissions

For more detailed troubleshooting, press `F1` in the application to access the comprehensive help system.

## 📝 Recent Updates

### Version 1.4.0 - Advanced Research & Comparison Edition

#### 🆕 Major New Features

- **Comparison Mode**: Complete image comparison system with quality metrics
- **Quality Metrics Integration**: PSNR, SSIM, MSE, LPIPS, and FID calculation
- **Batch Exhaustive Research**: Run parameter optimization across multiple image pairs
- **Advanced Metrics Visualization**: SSIM difference maps, LPIPS perceptual scores, and FID batch analysis
- **Comprehensive Reporting**: Export detailed comparison and research results

#### 🎯 Comparison Features

- **Side-by-Side Visualization**: Professional image comparison interface
- **Quality Metrics Calculation**: PSNR, SSIM, and LPIPS computation with quality interpretation
- **SSIM Difference Maps**: Heat map visualization of structural differences
- **Quality Assessment**: Automatic interpretation of metric values
- **Export Capabilities**: Save comparison reports and visualizations
- **Perceptual Metrics**: LPIPS for human-aligned quality assessment
- **Multi-format Support**: Works with all supported image formats

#### 🔬 Advanced Research Features

- **Batch Parameter Optimization**: Test multiple parameter combinations across image sets
- **Flexible Pair Selection**: Research all pairs, first N pairs, or manually selected pairs
- **Quality-based Optimization**: Find best parameters based on PSNR, SSIM, or LPIPS scores
- **FID Evaluation**: Assess distribution quality for large batches (50+ images)
- **Comprehensive Result Analysis**: CSV export with detailed metrics and timing data
- **Progress Monitoring**: Real-time tracking with detailed status information
- **Error Resilience**: Continue processing despite individual combination failures

#### 🏗️ Architecture Enhancements

- **New Models**: `ComparisonData` and `ImageMetrics` for quality analysis
- **New Controllers**: `ComparisonController` for comparison functionality
- **New Workers**: `BatchExhaustiveWorker` for advanced batch research
- **New UI Components**: `ComparisonPanel` and `BatchExhaustiveDialog` widgets
- **Enhanced Main Window**: Triple-mode interface supporting comparison mode

#### 🔧 Technical Improvements

- **Quality Metrics Library**: Integration with scikit-image for traditional metrics (PSNR, SSIM, MSE)
- **Deep Learning Metrics**: LPIPS implementation using PyTorch for perceptual similarity
- **Distribution Metrics**: FID calculation for batch quality assessment (50+ images)
- **Advanced Visualization**: SSIM difference maps and quality heat maps
- **Enhanced Export System**: Comprehensive reporting with multiple format support
- **Improved Error Handling**: Better validation and error recovery for comparison operations
- **GPU Acceleration**: LPIPS and FID support GPU acceleration when available
- **Memory Optimization**: Efficient handling of large image comparisons

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
- Scikit-image for traditional quality metrics (PSNR, SSIM, MSE)
- PyTorch for deep learning-based metrics (LPIPS, FID)
- Matplotlib and Seaborn for data visualization and analytics

---

**Note**: This application is designed for educational and research purposes. For commercial use, please ensure compliance with all relevant licenses and regulations. 
