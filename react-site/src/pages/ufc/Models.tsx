import { useState } from 'react';

export default function ModelsPage() {
  // Dummy data for the 6 models
  const dummyModels = [
    { name: 'AlgoPicks', accuracy: 85, color: 'from-orange-600 to-orange-400' },
    { name: 'XGBoost', accuracy: 82, color: 'from-blue-600 to-blue-400' },
    { name: 'Gradient', accuracy: 78, color: 'from-purple-600 to-purple-400' },
    { name: 'Logistic', accuracy: 75, color: 'from-green-600 to-green-400' },
    { name: 'Ensemble Weight Avg Prob', accuracy: 80, color: 'from-red-600 to-red-400' },
    { name: 'Ensemble Avg Prob', accuracy: 77, color: 'from-yellow-600 to-yellow-400' }
  ];

  return (
    <div className="w-full h-full flex items-end justify-around gap-8 p-8">
      {dummyModels.map((model) => (
        <div key={model.name} className="flex flex-col items-center flex-1 h-full">
          {/* Bar container */}
          <div className="w-full bg-slate-700/30 rounded-t-lg relative flex-1 flex items-end">
            <div 
              className={`w-full bg-gradient-to-t ${model.color} rounded-t-lg transition-all duration-500 hover:opacity-90 flex items-center justify-center`}
              style={{ height: `${model.accuracy}%` }}
            >
              <span className="text-white font-bold text-4xl drop-shadow-lg">{model.accuracy}%</span>
            </div>
          </div>
          {/* Label */}
          <p className="text-white font-semibold mt-4 text-center text-lg leading-tight">
            {model.name.includes('Ensemble') ? (
              <>
                {model.name.split(' ')[0]}<br/>{model.name.split(' ').slice(1).join(' ')}
              </>
            ) : (
              model.name
            )}
          </p>
        </div>
      ))}
    </div>
  );
}