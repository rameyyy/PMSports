interface DayStatsProps {
  modelRecord: { wins: number; losses: number };
  vegasRecord: { wins: number; losses: number };
  modelMAE: number;
  vegasMAE: number;
  avgConfidence: number;
  avgVig: number;
  edgeAfterVig: number;
}

export default function DayStats({ modelRecord, vegasRecord, modelMAE, vegasMAE, avgConfidence, avgVig, edgeAfterVig }: DayStatsProps) {
  const modelAccuracy = (modelRecord.wins / (modelRecord.wins + modelRecord.losses)) * 100;
  const vegasAccuracy = (vegasRecord.wins / (vegasRecord.wins + vegasRecord.losses)) * 100;
  const edge = modelAccuracy - vegasAccuracy;

  return (
    <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 mb-3">
      {/* Model ML */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Model ML</div>
        <div className="text-sm sm:text-lg font-bold text-white">{modelRecord.wins}-{modelRecord.losses}</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{modelAccuracy.toFixed(1)}%</div>
      </div>

      {/* Vegas ML */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Vegas ML</div>
        <div className="text-sm sm:text-lg font-bold text-white">{vegasRecord.wins}-{vegasRecord.losses}</div>
        <div className="text-[10px] sm:text-xs text-slate-500">{vegasAccuracy.toFixed(1)}%</div>
      </div>

      {/* Avg Model Conf */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Avg Model Conf</div>
        <div className="text-sm sm:text-lg font-bold text-white">{avgConfidence.toFixed(1)}%</div>
      </div>

      {/* Avg Vig */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Avg Vig</div>
        <div className="text-sm sm:text-lg font-bold text-white">{avgVig.toFixed(1)}%</div>
      </div>

      {/* Edge */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Edge</div>
        <div className={`text-sm sm:text-lg font-bold ${edge >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {edge >= 0 ? '+' : ''}{edge.toFixed(1)}%
        </div>
      </div>

      {/* After Vig */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Edge After Vig</div>
        <div className={`text-sm sm:text-lg font-bold ${edgeAfterVig >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {edgeAfterVig >= 0 ? '+' : ''}{edgeAfterVig.toFixed(1)}%
        </div>
      </div>

      {/* Model Totals MAE */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Model Totals MAE</div>
        <div className="text-sm sm:text-lg font-bold text-white">{modelMAE.toFixed(1)}</div>
      </div>

      {/* Vegas Totals MAE */}
      <div className="bg-slate-800/80 rounded-lg p-2 sm:p-3 text-center">
        <div className="text-[11px] sm:text-sm text-slate-400 uppercase tracking-wide">Vegas Totals MAE</div>
        <div className="text-sm sm:text-lg font-bold text-white">{vegasMAE.toFixed(1)}</div>
      </div>
    </div>
  );
}
