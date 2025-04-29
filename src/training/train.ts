import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as fs from 'fs';
import * as path from 'path';

// Acne types we want to classify
const ACNE_TYPES = [
  'whitehead',
  'blackhead',
  'papule',
  'pustule',
  'nodule',
  'cyst',
  'normal'
];

async function loadAndPreprocessImage(filePath: string): Promise<tf.Tensor3D> {
  const imageBuffer = fs.readFileSync(filePath);
  const imageTensor = tf.node.decodeImage(imageBuffer) as tf.Tensor3D;
  const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
  const normalizedImage = resizedImage.div(255.0);
  imageTensor.dispose();
  return normalizedImage;
}

async function loadDataset(dataDir: string): Promise<{ images: tf.Tensor3D[], labels: number[] }> {
  const images: tf.Tensor3D[] = [];
  const labels: number[] = [];

  for (const acneType of ACNE_TYPES) {
    const typeDir = path.join(dataDir, acneType);
    if (!fs.existsSync(typeDir)) {
      console.warn(`Directory ${typeDir} does not exist. Skipping...`);
      continue;
    }
    
    const files = fs.readdirSync(typeDir);
    for (const file of files) {
      if (file.endsWith('.jpg') || file.endsWith('.png')) {
        const imagePath = path.join(typeDir, file);
        const image = await loadAndPreprocessImage(imagePath);
        images.push(image);
        labels.push(ACNE_TYPES.indexOf(acneType));
      }
    }
  }

  return { images, labels };
}

async function createModel() {
  // Load MobileNet
  const baseModel = await mobilenet.load();
  const model = tf.sequential();

  // Add the base model (excluding the top layer)
  const baseModelOutput = (baseModel as any).model.layers[(baseModel as any).model.layers.length - 2].output;
  const newModel = tf.model({
    inputs: (baseModel as any).model.inputs,
    outputs: baseModelOutput
  });

  // Add new layers for acne classification
  model.add(tf.layers.inputLayer({ inputShape: [224, 224, 3] }));
  model.add(newModel);
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: ACNE_TYPES.length, activation: 'softmax' }));

  return model;
}

async function train() {
  console.log('Loading dataset...');
  const { images, labels } = await loadDataset(path.join(__dirname, 'data'));

  if (images.length === 0) {
    console.error('No images found in the dataset. Please add images to the data directory.');
    return;
  }

  console.log(`Loaded ${images.length} images for training`);
  console.log('Creating model...');
  const model = await createModel();

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Convert data to tensors
  const xs = tf.stack(images);
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), ACNE_TYPES.length);

  console.log('Training model...');
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
        }
      }
    }
  });

  // Save the model
  console.log('Saving model...');
  await model.save('file://./src/models/acne-classifier');

  // Clean up
  tf.dispose([xs, ys]);
  images.forEach(img => img.dispose());
}

train().catch(console.error); 