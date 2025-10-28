import { useState, useEffect } from 'react';
import type { ModelAccuracy } from './types';
import { fetchModelAccuracies } from './api';

export default function ModelsPage() {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [models, setModels] = useState<ModelAccuracy[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadModelAccuracies = async () => {
      try {
        const data = await fetchModelAccuracies();
        setModels(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching model accuracies:', error);
        setLoading(false);
      }
    };

    loadModelAccuracies();
  }, []);

  if (loading) {
    return (
      <div className="w-full p-4 sm:p-8 flex items-center justify-center">
        <p className="text-white text-lg">Loading model accuracies...</p>
      </div>
    );
  }

  // Find the best model (highest accuracy)
  const bestAccuracy = Math.max(...models.map(m => m.accuracy));

  const toggleModel = (modelName: string) => {
    setExpandedModel(expandedModel === modelName ? null : modelName);
  };

  return (
    <div className="w-full p-4 sm:p-8">
      <h1 className="text-lg sm:text-2xl font-bold text-white mb-6">Model Accuracies</h1>
      
      <div className="space-y-4">
        {models.map((model) => {
          const isBest = model.accuracy === bestAccuracy;
          const isExpanded = expandedModel === model.model_name;
          const incorrectPredictions = model.total_predictions - model.correct_predictions;
          
          return (
            <div key={model.model_name} className="w-full">
              {/* Model name and accuracy */}
              <div className="flex items-center justify-between mb-2">
                <h3 className={`text-sm sm:text-base font-semibold ${isBest ? 'text-orange-500' : 'text-white'}`}>
                  {model.model_name}
                </h3>
                <span className={`text-lg sm:text-xl font-bold ${isBest ? 'text-orange-500' : 'text-orange-400/70'}`}>
                  {model.accuracy.toFixed(1)}%
                </span>
              </div>
              
              {/* Horizontal bar - now clickable */}
              <button
                onClick={() => toggleModel(model.model_name)}
                className="w-full bg-slate-700/30 rounded-lg h-8 overflow-hidden cursor-pointer hover:bg-slate-700/40 transition-colors"
              >
                <div 
                  className={`h-full rounded-lg transition-all duration-500 flex items-center justify-start px-4 ${
                    isBest 
                      ? 'bg-gradient-to-r from-orange-500 to-orange-600' 
                      : 'bg-gradient-to-r from-orange-400/60 to-orange-500/60'
                  }`}
                  style={{ width: `${model.accuracy}%` }}
                >
                </div>
              </button>

              {/* Visual indicator */}
              <div className="flex items-center justify-center mt-2">
                <button
                  onClick={() => toggleModel(model.model_name)}
                  className="text-slate-400 text-xs hover:text-slate-300 transition-colors flex items-center gap-1"
                >
                  <span>{isExpanded ? 'Hide' : 'See'} Model Info</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>

              {/* Expanded stats dropdown */}
              {isExpanded && (
                <div className="mt-3 bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Total Predictions</p>
                      <p className="text-white text-sm sm:text-lg font-semibold">{model.total_predictions}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Correct Predictions</p>
                      <p className="text-green-400 text-sm sm:text-lg font-semibold">{model.correct_predictions}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Incorrect Predictions</p>
                      <p className="text-red-400 text-sm sm:text-lg font-semibold">{incorrectPredictions}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs mb-1">Avg Confidence</p>
                      <p className="text-white text-sm sm:text-lg font-semibold">{model.avg_confidence.toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}