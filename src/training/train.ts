import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// const ACNE_TYPES = ['whitehead', 'blackhead', 'papule', 'pustule', 'nodule', 'cyst', 'normal'];
const ACNE_TYPES = ['whitehead', 'blackhead', 'papule', 'pustule'];

async function loadAndPreprocessImage(filePath: string): Promise<tf.Tensor3D> {
  const buffer = fs.readFileSync(filePath);
  const imageTensor = tf.node.decodeImage(buffer, 3);
  const resized = tf.image.resizeBilinear(imageTensor as tf.Tensor3D, [224, 224]);
  return resized.div(255.0) as tf.Tensor3D;
}

async function loadDataset(dataDir: string, mobilenetModel: mobilenet.MobileNet) {
  const features: tf.Tensor2D[] = [];
  const labels: number[] = [];

  for (const acneType of ACNE_TYPES) {
    const typeDir = path.join(dataDir, acneType);
    if (!fs.existsSync(typeDir)) {
      console.warn(`⚠️  Skipping missing folder: ${typeDir}`);
      continue;
    }

    const files = fs.readdirSync(typeDir);
    for (const file of files) {
      if (!file.match(/\.(jpg|jpeg|png)$/i)) continue;
      try {
        const imgPath = path.join(typeDir, file);
        const image = await loadAndPreprocessImage(imgPath);
        const embedding = mobilenetModel.infer(image.expandDims(0), true) as tf.Tensor;
        features.push((embedding.squeeze() as tf.Tensor2D));
        labels.push(ACNE_TYPES.indexOf(acneType));
      } catch (e) {
        console.warn(`❌ Failed to process ${file}: ${e}`);
      }
    }
  }

  if (features.length === 0) throw new Error('❌ No valid images found.');

  const xs = tf.stack(features) as tf.Tensor2D;
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), ACNE_TYPES.length);

  return { xs, ys };
}

function createClassifierModel(inputDim: number): tf.LayersModel {
  const input = tf.input({ shape: [inputDim] });

  const dense1 = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input) as tf.SymbolicTensor;
  const bn1 = tf.layers.batchNormalization().apply(dense1) as tf.SymbolicTensor;
  const drop1 = tf.layers.dropout({ rate: 0.4 }).apply(bn1) as tf.SymbolicTensor;

  const dense2 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(drop1) as tf.SymbolicTensor;
  const bn2 = tf.layers.batchNormalization().apply(dense2) as tf.SymbolicTensor;
  const drop2 = tf.layers.dropout({ rate: 0.3 }).apply(bn2) as tf.SymbolicTensor;

  const dense3 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(drop2) as tf.SymbolicTensor;
  const bn3 = tf.layers.batchNormalization().apply(dense3) as tf.SymbolicTensor;

  const output = tf.layers.dense({ units: ACNE_TYPES.length, activation: 'softmax' }).apply(bn3) as tf.SymbolicTensor;

  return tf.model({ inputs: input, outputs: output });
}

async function ensureDirectoryExists(savePath: string) {
  if (!fs.existsSync(savePath)) {
    fs.mkdirSync(savePath, { recursive: true });
  }
}

async function main() {
  console.log('📦 Loading MobileNet...');
  const mobilenetModel = await mobilenet.load();
  console.log('✅ MobileNet loaded.');

  const dataDir = path.join(__dirname, 'data');
  const { xs, ys } = await loadDataset(dataDir, mobilenetModel);
  console.log(`✅ Loaded ${xs.shape[0]} samples.`);

  if (xs.shape[1] === undefined) throw new Error('❌ Feature dimension is undefined.');
  const model = createClassifierModel(xs.shape[1]);
  const optimizer = tf.train.adam(0.0003);

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  console.log('🚀 Training model...');

  await model.fit(xs, ys, {
    epochs: 40,
    batchSize: 4,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const valLoss = logs?.val_loss ?? Infinity;
        const valAcc = logs?.val_acc ?? 0;
        console.log(`Epoch ${epoch + 1} - loss: ${logs?.loss?.toFixed(4)}, acc: ${logs?.acc?.toFixed(4)}, val_acc: ${valAcc.toFixed(4)}, val_loss: ${valLoss.toFixed(4)}`);
      },
      onTrainEnd: () => console.log('✅ Training complete')
    },
  });

  const modelPath = path.join(__dirname, '../../public/models/acne-classifier');
  await ensureDirectoryExists(modelPath);

  console.log('💾 Saving model to', modelPath);
  await model.save(`file://${modelPath}`);

  tf.dispose([xs, ys]);
  console.log('🏁 Training finished.');
}

main().catch(console.error);
