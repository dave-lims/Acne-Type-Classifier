'use client';

import { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { useSkinAnalysis } from '@/hooks/useSkinAnalysis';

export default function Home() {
  const webcamRef = useRef<Webcam>(null);
  const [image, setImage] = useState<string | null>(null);
  const { loading, error, predictions, analyzeImage } = useSkinAnalysis();

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setImage(imageSrc);
      const img = new Image();
      img.src = imageSrc;
      img.onload = () => analyzeImage(img);
    }
  }, [analyzeImage]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageSrc = e.target?.result as string;
        setImage(imageSrc);
        const img = new Image();
        img.src = imageSrc;
        img.onload = () => analyzeImage(img);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Skin Analysis App</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="border-2 border-gray-300 rounded-lg overflow-hidden">
              {image ? (
                <img src={image} alt="Captured" className="w-full h-auto" />
              ) : (
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="w-full h-auto"
                />
              )}
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={capture}
                className="flex-1 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Take Photo
              </button>
              <label className="flex-1 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 text-center cursor-pointer">
                Upload Photo
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          <div className="space-y-4">
            <h2 className="text-2xl font-semibold">Analysis Results</h2>
            {loading && <p>Loading model...</p>}
            {error && <p className="text-red-500">{error}</p>}
            {predictions.length > 0 && (
              <div className="space-y-2">
                {predictions.map((prediction, index) => (
                  <div key={index} className="bg-gray-100 p-4 rounded">
                    <p className="font-medium">{prediction.className}</p>
                    <p>Confidence: {(prediction.probability * 100).toFixed(2)}%</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
