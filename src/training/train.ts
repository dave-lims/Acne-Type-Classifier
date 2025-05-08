import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ACNE_TYPES = ['whitehead', 'blackhead', 'papule', 'pustule', 'nodule', 'cyst', 'normal'];

async function loadAndPreprocessImage(filePath: string): Promise<tf.Tensor3D> {
  const buffer = fs.readFileSync(filePath);
  const imageTensor = tf.node.decodeImage(buffer, 3) as tf.Tensor3D;
  const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
  return resized.div(255.0) as tf.Tensor3D;
}

async function loadDataset(dataDir: string, mobilenetModel: mobilenet.MobileNet) {
  const features: tf.Tensor2D[] = [];
  const labels: number[] = [];

  for (const acneType of ACNE_TYPES) {
    const typeDir = path.join(dataDir, acneType);
    if (!fs.existsSync(typeDir)) {
      console.warn(`⚠️  Missing directory: ${typeDir}`);
      continue;
    }

    const files = fs.readdirSync(typeDir);
    for (const file of files) {
      if (!file.match(/\.(jpg|jpeg|png)$/i)) continue;

      const imgPath = path.join(typeDir, file);
      const image = await loadAndPreprocessImage(imgPath);
      const embedding = mobilenetModel.infer(image.expandDims(0), true) as tf.Tensor;
      features.push(embedding.squeeze() as tf.Tensor2D);
      labels.push(ACNE_TYPES.indexOf(acneType));
    }
  }

  if (features.length === 0) throw new Error('❌ No valid images found in dataset.');

  const xs = tf.stack(features) as tf.Tensor2D;
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), ACNE_TYPES.length) as tf.Tensor2D;

  return { xs, ys };
}

function createClassifierModel(inputDim: number): tf.LayersModel {
  const input = tf.input({ shape: [inputDim] });
  const dense1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(input) as tf.SymbolicTensor;
  const dropout = tf.layers.dropout({ rate: 0.3 }).apply(dense1) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: ACNE_TYPES.length, activation: 'softmax' }).apply(dropout) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: output });
}

async function train(): Promise<void> {
  console.log('📦 Loading MobileNet...');
  const mobilenetModel = await mobilenet.load();
  console.log('✅ MobileNet loaded.');

  const dataDir = path.join(__dirname, 'data');
  console.log('📂 Loading dataset...');
  const { xs, ys } = await loadDataset(dataDir, mobilenetModel);
  console.log(`✅ Loaded ${xs.shape[0]} samples`);

  const model = createClassifierModel(xs.shape[1]);

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  console.log('🚀 Training model...');
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} - loss: ${logs?.loss?.toFixed(4)}, acc: ${logs?.acc?.toFixed(4)}`);
      },
    },
  });

  const modelPath = path.join(__dirname, '../models/acne-classifier');
  console.log('💾 Saving model to', modelPath);
  await model.save(`file://${modelPath}`);

  tf.dispose([xs, ys]);
}

train().catch(console.error);
