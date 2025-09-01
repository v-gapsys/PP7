# COVID-19 X-ray Classification with PyTorch

A complete deep learning pipeline for COVID-19 detection using chest X-ray images from the COVID-Xray-5k dataset, optimized for Apple Silicon with Metal Performance Shaders (MPS).

## Features

- **Metal Performance Shaders (MPS)** support for Apple Silicon
- **Complete training pipeline** with validation and testing
- **Real-time visualization** of training progress
- **Comprehensive evaluation** with detailed metrics
- **Model saving and loading** functionality
- **Data leakage prevention** with proper train/test splits

## Dataset

This project uses the **COVID-Xray-5k dataset** which contains:
- **Training set**: COVID-19 and non-COVID chest X-ray images
- **Test set**: Separate, unseen COVID-19 and non-COVID images
- **Patient-level splitting** to prevent data leakage

## Requirements

- Python 3.8+
- PyTorch 2.8.0+
- macOS with Apple Silicon (for MPS support) or CUDA-capable GPU

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid-xray-classification.git
cd covid-xray-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Download the dataset** by running the notebook cells (automatic download from Dropbox)
2. **Run the training pipeline** by executing all cells in `covid_xray_5k_full_training.ipynb`
3. **Monitor training progress** with real-time visualizations
4. **Evaluate results** on the test set

## Model Architecture

- **Backbone**: ResNet18 pre-trained on ImageNet
- **Modifications**: 
  - First layer adapted for grayscale X-ray images
  - Custom classification head with dropout layers
  - Optimized for Metal performance on Apple Silicon

## Training Configuration

- **Batch size**: 64 (MPS) / 32 (CPU/CUDA)
- **Learning rate**: 0.001
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 30

## Results

The model achieves high accuracy on the test set with proper patient-level splitting to ensure reliable evaluation metrics.

## File Structure

```
├── covid_xray_5k_full_training.ipynb    # Main training notebook
├── covid_xray_5k_full_training_clean.ipynb  # Clean version without emojis
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── .gitignore                           # Git ignore file
└── venv/                                # Virtual environment (not tracked)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- COVID-Xray-5k dataset creators
- PyTorch team for MPS support
- Apple for Metal Performance Shaders

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{covid_xray_classification_2024,
  title={COVID-19 X-ray Classification with PyTorch and Metal Performance Shaders},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/covid-xray-classification}
}
```
