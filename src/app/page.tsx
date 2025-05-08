"use client";

import { useState, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import { useSkinAnalysis } from "@/hooks/useSkinAnalysis";

export default function Home() {
  const webcamRef = useRef<Webcam>(null);
  const { loading, error, currentAnalysis, analyzeImage } = useSkinAnalysis();
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const capture = useCallback(async () => {
    if (webcamRef.current) {
      setIsAnalyzing(true);
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const img = new Image();
        img.src = imageSrc;
        img.onload = () => {
          analyzeImage(img);
          setIsAnalyzing(false);
        };
      }
    }
  }, [analyzeImage]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageSrc = e.target?.result as string;
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
        <h1 className="text-4xl font-bold text-center mb-8">
          Skin Analysis App
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="border-2 border-gray-300 rounded-lg overflow-hidden">
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full h-auto"
                style={{ transform: "scaleX(-1)" }}
              />
            </div>

            <div className="flex gap-4">
              <button
                onClick={capture}
                disabled={loading || isAnalyzing}
                className="flex-1 bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
              >
                {isAnalyzing ? "Analyzing..." : "Capture & Analyze"}
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
            {currentAnalysis.topPrediction && (
              <div className="space-y-4">
                <div className="bg-green-100 p-4 rounded">
                  <h2 className="font-bold">Top Prediction:</h2>
                  <p>
                    {currentAnalysis.topPrediction.className} (
                    {currentAnalysis.topPrediction.probability.toFixed(1)}%
                    confidence)
                  </p>
                </div>

                <div className="space-y-2">
                  <h2 className="font-bold">All Predictions:</h2>
                  {currentAnalysis.allPredictions.map((prediction, index) => (
                    <div key={index} className="bg-gray-100 p-4 rounded">
                      <p>
                        {prediction.className}:{" "}
                        {prediction.probability.toFixed(1)}%
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
