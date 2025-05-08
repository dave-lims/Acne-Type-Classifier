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
  const [classifierModel, setClassifierModel] = useState<tf.LayersModel | null>(null);
  const [mobilenetModel, setMobilenetModel] = useState<mobilenet.MobileNet | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentAnalysis, setCurrentAnalysis] = useState<{
    topPrediction: { className: string; probability: number } | null;
    allPredictions: Array<{ className: string; probability: number }>;
  }>({
    topPrediction: null,
    allPredictions: []
  });

  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();
        // Load the fine-tuned classifier model
        const loadedClassifier = await tf.loadLayersModel('/models/acne-classifier/model.json');
        setClassifierModel(loadedClassifier);
        // Load MobileNet for feature extraction
        const loadedMobileNet = await mobilenet.load();
        setMobilenetModel(loadedMobileNet);
        setLoading(false);
      } catch (err) {
        setError('Failed to load the models');
        setLoading(false);
      }
    };
    loadModels();
  }, []);

  const clearAnalysis = useCallback(() => {
    setCurrentAnalysis({
      topPrediction: null,
      allPredictions: []
    });
    setError(null);
  }, []);

  const analyzeImage = useCallback(async (image: HTMLImageElement | HTMLVideoElement) => {
    if (!classifierModel || !mobilenetModel) {
      setError('Models not loaded');
      return;
    }

    try {
      clearAnalysis();
      // Preprocess the image
      const tensor = tf.browser.fromPixels(image)
        .resizeBilinear([224, 224])
        .div(255.0)
        .expandDims(0);

      // Extract features using MobileNet
      const features = mobilenetModel.infer(tensor, true) as tf.Tensor;
      // Pass features to classifier model
      const predictions = await classifierModel.predict(features) as tf.Tensor;
      const data = await predictions.data();

      // Convert predictions to readable format
      const results = Array.from(data).map((probability, index) => {
        const className = ACNE_TYPES[index];
        const confidence = Math.min(Math.max(probability * 100, 0), 100);
        return {
          className,
          probability: confidence
        };
      });

      // Sort by probability
      results.sort((a, b) => b.probability - a.probability);

      setCurrentAnalysis({
        topPrediction: results[0],
        allPredictions: results
      });

      tensor.dispose();
      features.dispose();
      predictions.dispose();
    } catch (err) {
      setError('Failed to analyze image');
    }
  }, [classifierModel, mobilenetModel, clearAnalysis]);

  return {
    model: classifierModel,
    loading,
    error,
    currentAnalysis,
    analyzeImage,
    clearAnalysis
  };
}; 