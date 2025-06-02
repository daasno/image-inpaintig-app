"""
Control panel widget containing all inpainting parameters and settings
"""
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                               QRadioButton, QSpinBox, QDoubleSpinBox, QButtonGroup)
from PySide6.QtCore import Signal

from models.inpaint_worker import InpaintWorker
from config.settings import AppConstants


class ControlPanel(QGroupBox):
    """Control panel for inpainting parameters"""
    
    # Signals
    implementation_changed = Signal(str)  # "CPU" or "GPU"
    patch_size_changed = Signal(int)
    p_value_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__("Inpainting Controls", parent)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QGroupBox {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #555;
                border-radius: 6px;
                margin-top: 1ex;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #ffffff;
                background-color: #2b2b2b;
            }
            QLabel {
                color: #cccccc;
                background: transparent;
            }
            QRadioButton {
                color: #cccccc;
                background: transparent;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #666;
                border-radius: 7px;
                background-color: #3a3a3a;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #007acc;
                border-radius: 7px;
                background-color: #007acc;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px;
                color: #cccccc;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #007acc;
            }
        """)
        
        self.setup_ui()
        self.connect_signals()
        self.update_gpu_availability()
    
    def setup_ui(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout(self)
        
        # Implementation selection
        impl_layout = QHBoxLayout()
        impl_label = QLabel("Implementation:")
        
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        
        # Group the radio buttons
        self.impl_group = QButtonGroup(self)
        self.impl_group.addButton(self.cpu_radio)
        self.impl_group.addButton(self.gpu_radio)
        
        impl_layout.addWidget(impl_label)
        impl_layout.addWidget(self.cpu_radio)
        impl_layout.addWidget(self.gpu_radio)
        impl_layout.addStretch()
        
        layout.addLayout(impl_layout)
        
        # Patch size control
        patch_layout = QHBoxLayout()
        patch_label = QLabel("Patch Size:")
        
        self.patch_size_spinner = QSpinBox()
        self.patch_size_spinner.setRange(AppConstants.MIN_PATCH_SIZE, AppConstants.MAX_PATCH_SIZE)
        self.patch_size_spinner.setValue(9)
        self.patch_size_spinner.setSingleStep(2)  # Only allow odd numbers
        self.patch_size_spinner.setToolTip("Size of the patch used for matching (must be odd)")
        
        patch_layout.addWidget(patch_label)
        patch_layout.addWidget(self.patch_size_spinner)
        patch_layout.addStretch()
        
        layout.addLayout(patch_layout)
        
        # Minkowski order parameter control
        p_layout = QHBoxLayout()
        p_label = QLabel("Minkowski Order (p):")
        
        self.p_value_spinner = QDoubleSpinBox()
        self.p_value_spinner.setRange(AppConstants.MIN_P_VALUE, AppConstants.MAX_P_VALUE)
        self.p_value_spinner.setValue(1.0)
        self.p_value_spinner.setSingleStep(0.1)
        self.p_value_spinner.setDecimals(1)
        self.p_value_spinner.setToolTip("Distance metric: 1.0=Manhattan, 2.0=Euclidean")
        
        p_layout.addWidget(p_label)
        p_layout.addWidget(self.p_value_spinner)
        p_layout.addStretch()
        
        layout.addLayout(p_layout)
    
    def connect_signals(self):
        """Connect widget signals to methods"""
        self.cpu_radio.toggled.connect(self._on_implementation_changed)
        self.gpu_radio.toggled.connect(self._on_implementation_changed)
        self.patch_size_spinner.valueChanged.connect(self._on_patch_size_changed)
        self.p_value_spinner.valueChanged.connect(self._on_p_value_changed)
    
    def update_gpu_availability(self):
        """Update GPU availability and set default selection"""
        gpu_available = InpaintWorker.check_gpu_availability()
        
        if gpu_available:
            self.gpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(True)
            self.gpu_radio.setToolTip("GPU acceleration available")
        else:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)
            
            # Check why GPU is not available
            if not InpaintWorker.check_gpu_availability():
                try:
                    import numba
                    if not numba.cuda.is_available():
                        self.gpu_radio.setToolTip("CUDA is not available on this system")
                    else:
                        self.gpu_radio.setToolTip("GPU implementation not found")
                except ImportError:
                    self.gpu_radio.setToolTip("Numba CUDA is not installed")
            else:
                self.gpu_radio.setToolTip("GPU implementation not available")
    
    def _on_implementation_changed(self):
        """Handle implementation radio button changes"""
        if self.cpu_radio.isChecked():
            self.implementation_changed.emit("CPU")
        elif self.gpu_radio.isChecked():
            self.implementation_changed.emit("GPU")
    
    def _on_patch_size_changed(self, value):
        """Handle patch size changes and ensure odd values"""
        # Force odd numbers only
        if value % 2 == 0:
            self.patch_size_spinner.setValue(value + 1)
            return
        
        self.patch_size_changed.emit(value)
    
    def _on_p_value_changed(self, value):
        """Handle p-value changes"""
        self.p_value_changed.emit(value)
    
    def get_current_implementation(self) -> str:
        """Get the currently selected implementation"""
        return "GPU" if self.gpu_radio.isChecked() else "CPU"
    
    def get_current_patch_size(self) -> int:
        """Get the current patch size"""
        return self.patch_size_spinner.value()
    
    def get_current_p_value(self) -> float:
        """Get the current p-value"""
        return self.p_value_spinner.value()
    
    def set_implementation(self, implementation: str):
        """Set the implementation selection"""
        if implementation.upper() == "GPU" and self.gpu_radio.isEnabled():
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
    
    def set_patch_size(self, size: int):
        """Set the patch size"""
        # Ensure it's odd
        if size % 2 == 0:
            size += 1
        
        size = max(AppConstants.MIN_PATCH_SIZE, min(AppConstants.MAX_PATCH_SIZE, size))
        self.patch_size_spinner.setValue(size)
    
    def set_p_value(self, value: float):
        """Set the p-value"""
        value = max(AppConstants.MIN_P_VALUE, min(AppConstants.MAX_P_VALUE, value))
        self.p_value_spinner.setValue(value)
    
    def get_parameters(self) -> dict:
        """Get all current parameters as a dictionary"""
        return {
            'implementation': self.get_current_implementation(),
            'patch_size': self.get_current_patch_size(),
            'p_value': self.get_current_p_value()
        }
    
    def set_parameters(self, params: dict):
        """Set parameters from a dictionary"""
        if 'implementation' in params:
            self.set_implementation(params['implementation'])
        if 'patch_size' in params:
            self.set_patch_size(params['patch_size'])
        if 'p_value' in params:
            self.set_p_value(params['p_value'])
    
    def validate_parameters(self) -> tuple[bool, str]:
        """Validate current parameters"""
        implementation = self.get_current_implementation()
        
        # Check implementation availability
        if implementation == "GPU" and not InpaintWorker.check_gpu_availability():
            return False, "GPU implementation is not available"
        
        if implementation == "CPU" and not InpaintWorker.check_cpu_availability():
            return False, "CPU implementation is not available"
        
        # Check patch size
        patch_size = self.get_current_patch_size()
        if patch_size % 2 == 0:
            return False, "Patch size must be odd"
        
        if patch_size < AppConstants.MIN_PATCH_SIZE or patch_size > AppConstants.MAX_PATCH_SIZE:
            return False, f"Patch size must be between {AppConstants.MIN_PATCH_SIZE} and {AppConstants.MAX_PATCH_SIZE}"
        
        # Check p-value
        p_value = self.get_current_p_value()
        if p_value < AppConstants.MIN_P_VALUE or p_value > AppConstants.MAX_P_VALUE:
            return False, f"P-value must be between {AppConstants.MIN_P_VALUE} and {AppConstants.MAX_P_VALUE}"
        
        return True, "" 