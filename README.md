# Background Blur App with Deep Learning

A Python application that uses PyTorch and DeepLabV3 to automatically detect subjects in photos and blur the background while keeping the foreground sharp.

## Features

- **Automatic subject detection** using DeepLabV3 semantic segmentation
- **Adjustable blur strength** for customized background effects
- **Multiple interface options**: CLI, Flask web app, and Streamlit UI
- **Fast processing** leveraging GPU acceleration (when available)
  
### Prerequisites

- Python 3.7+
- PyTorch (with CUDA if available)
- Basic system dependencies (for image processing):

### Setup

Clone The Repository
- git clone https://github.com/YASHMANIC/background_blur.git
- cd background_blur
  
Install Dependencies
- pip install -r requirements.txt

Web Interface (Flask)
- python app.py
- Then open http://localhost:5000 in your browser
