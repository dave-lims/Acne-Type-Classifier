# Skin Analysis App

A web application that uses AI to analyze skin conditions and classify different types of acne. The app uses TensorFlow.js and MobileNet for image classification.

## Features

- Real-time webcam capture for skin analysis
- Image upload functionality
- AI-powered acne classification
- Support for multiple acne types:
  - Whiteheads
  - Blackheads
  - Papules
  - Pustules
  - Nodules
  - Cysts
  - Normal skin

## Prerequisites

- Node.js (v18 or later)
- npm or yarn
- Webcam (for real-time analysis)

## Installation

1. Clone the repository:
```bash
https://github.com/jsong1004/Acne-Type-Classifier.git
cd Acne-Tyoe-Classifier
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Training the Model

To train the model with your own dataset, follow these steps:

1. Prepare your dataset:
   - Create the following directory structure in `src/training/data/`:
     ```
     src/training/data/
     ├── whitehead/
     ├── blackhead/
     ├── papule/
     ├── pustule/
     ├── nodule/
     ├── cyst/
     └── normal/
     ```
   - Add images to each directory (recommended: 50-100 images per category)
   - Image requirements:
     - High-quality, well-lit photos
     - Clear view of the affected area
     - Consistent image sizes (will be resized to 224x224)
     - Various skin tones and lighting conditions

2. Install training dependencies:
```bash
npm install @tensorflow/tfjs-node @types/node
```

3. Run the training script:
```bash
npx ts-node src/training/train.ts
```

The training process will:
- Load and preprocess your images
- Fine-tune the MobileNet model
- Save the trained model to `src/models/acne-classifier/`

Training parameters:
- Epochs: 10
- Batch size: 32
- Validation split: 20%
- Learning rate: 0.001

## Usage

1. Open the application in your web browser
2. Choose one of two options:
   - Click "Take Photo" to capture an image using your webcam
   - Click "Upload Photo" to select an image from your device
3. The app will analyze the image and display:
   - The detected acne type
   - Confidence level for each prediction

## Model Architecture

The application uses a fine-tuned MobileNet model:
1. Base MobileNet model for feature extraction
2. Additional dense layers for acne classification
3. Dropout layer to prevent overfitting
4. Softmax activation for multi-class classification

## Troubleshooting

If you encounter issues:
1. Ensure your webcam is properly connected and accessible
2. Check that images are in supported formats (JPG, PNG)
3. Verify sufficient lighting for accurate analysis
4. Clear browser cache if the model fails to load

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
