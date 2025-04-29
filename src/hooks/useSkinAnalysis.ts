import { useState, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

const ACNE_TYPES = [
  'whitehead',
  'blackhead',
  'papule',
  'pustule',
  'nodule',
  'cyst',
  'normal'
];

export const useSkinAnalysis = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Array<{ className: string; probability: number }>>([]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        // Try to load the fine-tuned model first
        try {
          const loadedModel = await tf.loadLayersModel('/models/acne-classifier/model.json');
          setModel(loadedModel);
        } catch (err) {
          console.warn('Could not load fine-tuned model, falling back to MobileNet');
          const baseModel = await mobilenet.load();
          setModel((baseModel as any).model);
        }
        setLoading(false);
      } catch (err) {
        setError('Failed to load the model');
        setLoading(false);
      }
    };

    loadModel();
  }, []);

  const analyzeImage = useCallback(async (image: HTMLImageElement | HTMLVideoElement) => {
    if (!model) {
      setError('Model not loaded');
      return;
    }

    try {
      // Preprocess the image
      const tensor = tf.browser.fromPixels(image)
        .resizeBilinear([224, 224])
        .div(255.0)
        .expandDims(0);

      // Make prediction
      const predictions = await model.predict(tensor) as tf.Tensor;
      const data = await predictions.data();
      
      // Convert predictions to readable format
      const results = Array.from(data).map((probability, index) => ({
        className: ACNE_TYPES[index],
        probability
      }));

      // Sort by probability
      results.sort((a, b) => b.probability - a.probability);
      setPredictions(results);

      // Clean up
      tensor.dispose();
      predictions.dispose();
    } catch (err) {
      setError('Failed to analyze image');
    }
  }, [model]);

  return {
    model,
    loading,
    error,
    predictions,
    analyzeImage,
  };
}; 