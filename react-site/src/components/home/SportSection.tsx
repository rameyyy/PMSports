import { useNavigate } from 'react-router-dom';

interface AccuracyData {
  winRate: string;
  record: string;
  totalPicks: number;
}

interface EventData {
  name: string;
  date: string;
  daysAway?: number;
  gamesCount?: number;
}

interface PickData {
  record: string;
  winRate: string;
  avgOdds?: string;
  roi?: string;
  todayPick?: {
    title: string;
    subtitle?: string;
    prediction: string;
    odds?: string;
    result: null;
  } | null;
  lastPick?: {
    title: string;
    subtitle?: string;
    prediction: string;
    odds?: string;
    result: 'correct' | 'incorrect' | null;
  } | null;
  pickLabel: string;
}

interface SportSectionProps {
  name: string;
  subtitle: string;
  logo: string;
  path: string;
  available: boolean;
  nextEvent: EventData;
  modelAccuracy: AccuracyData;
  vegasAccuracy: AccuracyData;
  edge: string;
  pick?: PickData;
  pickTitle?: string; // "Pick of the Day" or "Pick of the Week"
}

export default function SportSection({
  name,
  subtitle,
  logo,
  path,
  available,
  nextEvent,
  modelAccuracy,
  vegasAccuracy,
  edge,
  pick,
  pickTitle
}: SportSectionProps) {
  const navigate = useNavigate();

  const ResultIcon = ({ result }: { result: 'correct' | 'incorrect' | null }) => {
    if (result === null) return null;
    return result === 'correct' ? (
      <div className="w-7 h-7 rounded-full bg-green-400/20 flex items-center justify-center">
        <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      </div>
    ) : (
      <div className="w-7 h-7 rounded-full bg-red-400/20 flex items-center justify-center">
        <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </div>
    );
  };

  const content = (
    <>
      {/* Sport Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-700 group-hover:border-orange-500 transition-colors">
        <div className="flex items-center gap-3 md:gap-4">
          <img src={logo} alt={name} className="h-8 w-8 md:h-10 md:w-10 object-contain flex-shrink-0" />
          <div>
            <h2 className="text-xl md:text-2xl font-bold text-white group-hover:text-orange-500 transition-colors">
              {name}
            </h2>
            <p className="text-xs md:text-sm text-slate-400">{subtitle}</p>
          </div>
        </div>
        <svg className="w-6 h-6 text-slate-500 group-hover:text-orange-500 group-hover:translate-x-1 transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 pl-0 md:pl-14">
        {/* Next Event */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">
            {nextEvent.gamesCount ? "Today's Games" : "Next Event"}
          </p>
          <p className="text-white font-semibold">
            {nextEvent.gamesCount ? `${nextEvent.gamesCount} matchups` : nextEvent.name}
          </p>
          <p className="text-sm text-slate-400">
            {nextEvent.daysAway ? `${nextEvent.date} (${nextEvent.daysAway}d)` : nextEvent.date}
          </p>
        </div>

        {/* Model Accuracy */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">My Model</p>
          <p className="text-2xl font-bold text-white">{modelAccuracy.winRate}</p>
          <p className="text-sm text-slate-400">{modelAccuracy.record}</p>
        </div>

        {/* Vegas Accuracy */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Vegas Favorites</p>
          <p className="text-2xl font-bold text-white">{vegasAccuracy.winRate}</p>
          <p className="text-sm text-slate-400">{vegasAccuracy.record}</p>
        </div>

        {/* Edge */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Edge vs Vegas</p>
          <p className={`text-2xl font-bold ${parseFloat(edge) > 0 ? 'text-green-400' : 'text-orange-400'}`}>
            {edge}
          </p>
          <p className="text-sm text-slate-400">Accuracy advantage</p>
        </div>
      </div>

      {pick && pickTitle && (
        <div className="mt-6 pl-0 md:pl-14">
          <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">
            <div className="mb-3 flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-white">{pickTitle}</p>
                <p className="text-xs text-slate-400">{pick.record} ({pick.winRate})</p>
              </div>
              {(pick.avgOdds || pick.roi) && (
                <div className="text-right">
                  {pick.avgOdds && (
                    <p className="text-xs text-slate-400">Avg Odds: <span className="text-white font-medium">{pick.avgOdds}</span></p>
                  )}
                  {pick.roi && (
                    <p className={`text-xs font-semibold ${parseFloat(pick.roi) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ROI: {pick.roi}
                    </p>
                  )}
                </div>
              )}
            </div>

            {pick.todayPick && (
              <div className="mb-4 pb-4 border-b border-slate-700/50">
                <p className="text-xs text-slate-500 mb-1">Today's Pick</p>
                <p className="text-white font-medium text-sm mb-0.5">{pick.todayPick.title}</p>
                {pick.todayPick.subtitle && (
                  <p className="text-xs text-slate-400 mb-1">{pick.todayPick.subtitle}</p>
                )}
                <div className="flex items-center gap-2">
                  <p className="text-xs text-slate-300">Picked: {pick.todayPick.prediction}</p>
                  {pick.todayPick.odds && (
                    <span className="text-xs text-orange-400 font-medium">({pick.todayPick.odds})</span>
                  )}
                </div>
              </div>
            )}

            {pick.lastPick && (
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-xs text-slate-500 mb-1">{pick.pickLabel}</p>
                  <p className="text-white font-medium text-sm mb-0.5">{pick.lastPick.title}</p>
                  {pick.lastPick.subtitle && (
                    <p className="text-xs text-slate-400 mb-1">{pick.lastPick.subtitle}</p>
                  )}
                  <div className="flex items-center gap-2">
                    <p className="text-xs text-slate-300">Picked: {pick.lastPick.prediction}</p>
                    {pick.lastPick.odds && (
                      <span className="text-xs text-orange-400 font-medium">({pick.lastPick.odds})</span>
                    )}
                  </div>
                </div>
                {pick.lastPick.result && (
                  <div className="flex-shrink-0 ml-3">
                    <ResultIcon result={pick.lastPick.result} />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );

  return (
    <div className="group">
      {available ? (
        <button onClick={() => navigate(path)} className="w-full text-left">
          {content}
        </button>
      ) : (
        <div className="w-full text-left">
          {content}
        </div>
      )}
    </div>
  );
}
