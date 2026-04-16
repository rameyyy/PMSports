import { useState, useEffect } from 'react';
import { fetchPerformance, type BookPerformance } from '../../../api/ncaamb';

const BOOK_DISPLAY: Record<string, string> = {
  draftkings: 'DraftKings',
  fanduel: 'FanDuel',
  betmgm: 'BetMGM',
  bovada: 'Bovada',
  betonlineag: 'BetOnline',
  lowvig: 'LowVig',
  AlgoPicks: 'AlgoPicks',
};

function displayName(book: string) {
  return BOOK_DISPLAY[book] || book;
}

function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) return <span className="text-xs font-bold text-orange-400 bg-orange-500/15 px-1.5 py-0.5 rounded">1st</span>;
  if (rank === 2) return <span className="text-xs font-medium text-slate-300 bg-slate-600/40 px-1.5 py-0.5 rounded">2nd</span>;
  if (rank === 3) return <span className="text-xs font-medium text-slate-400 bg-slate-600/30 px-1.5 py-0.5 rounded">3rd</span>;
  return <span className="text-xs text-slate-500 px-1.5 py-0.5">{rank}th</span>;
}

export default function ModelPerformance() {
  const [books, setBooks] = useState<BookPerformance[]>([]);
  const [date, setDate] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPerformance()
      .then((data) => {
        setBooks(data.books);
        setDate(data.date);
      })
      .catch(() => setBooks([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-center text-slate-400 py-12">
        <div className="animate-spin inline-block w-6 h-6 border-2 border-slate-400 border-t-transparent rounded-full mb-2" />
        <p>Loading performance data...</p>
      </div>
    );
  }

  if (books.length === 0) {
    return (
      <div className="text-center text-slate-400 py-12">
        <p>No performance data available yet.</p>
      </div>
    );
  }

  const sportsbooks = books.filter((b) => b.book !== 'AlgoPicks');

  // Sort by ML accuracy desc
  const allByMlAcc = [...books]
    .filter((b) => b.ml_total > 0)
    .sort((a, b) => b.ml_right / b.ml_total - a.ml_right / a.ml_total);

  // Sort by O/U MAE asc (lower is better)
  const allByOuMae = [...books]
    .filter((b) => b.ou_mae !== null)
    .sort((a, b) => a.ou_mae! - b.ou_mae!);

  // Sort sportsbooks by AP O/U acc desc
  const booksByApOuAcc = [...sportsbooks]
    .filter((b) => b.ap_ou_acc !== null)
    .sort((a, b) => b.ap_ou_acc! - a.ap_ou_acc!);

  // Derive season year: Nov/Dec = next year, otherwise current year
  const dateObj = date ? new Date(date + 'T00:00:00') : new Date();
  const month = dateObj.getMonth() + 1;
  const seasonYear = month >= 11 ? dateObj.getFullYear() + 1 : dateObj.getFullYear();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-lg font-semibold text-white">{seasonYear} Season</h2>
      </div>

      {/* ML Accuracy Section */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-base font-semibold text-white">Moneyline Accuracy</h3>
          <p className="text-xs text-slate-500 mt-0.5">How often each source correctly picks the winner</p>
        </div>

        <div className="p-3 sm:p-4">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 text-xs uppercase tracking-wider border-b border-slate-700/50">
                  <th className="text-left pb-2.5 pl-2 font-medium">Rank</th>
                  <th className="text-left pb-2.5 font-medium">Source</th>
                  <th className="text-center pb-2.5 font-medium">Accuracy</th>
                  <th className="text-center pb-2.5 font-medium">Record</th>
                </tr>
              </thead>
              <tbody>
                {allByMlAcc.map((b, i) => {
                  const acc = (b.ml_right / b.ml_total) * 100;
                  const isAlgo = b.book === 'AlgoPicks';
                  return (
                    <tr key={b.book} className={`border-b border-slate-700/30 ${i % 2 === 0 ? 'bg-slate-700/10' : ''}`}>
                      <td className="py-2.5 pl-2">
                        <RankBadge rank={i + 1} />
                      </td>
                      <td className="py-2.5">
                        <span className={`font-medium ${isAlgo ? 'text-orange-400' : 'text-slate-200'}`}>
                          {displayName(b.book)}
                        </span>
                        {i === 0 && <span className="text-orange-400 ml-1">&#9733;</span>}
                      </td>
                      <td className="py-2.5 text-center">
                        <span className="text-white font-semibold tabular-nums">{acc.toFixed(1)}%</span>
                      </td>
                      <td className="py-2.5 text-center text-slate-400 tabular-nums">
                        {b.ml_right}-{b.ml_total - b.ml_right}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* O/U MAE Section */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-base font-semibold text-white">Total Point Prediction Accuracy (MAE)</h3>
          <p className="text-xs text-slate-500 mt-0.5">Mean Absolute Error &mdash; <span className="text-slate-400 font-mono">avg( |predicted - actual| )</span> &mdash; lower is better</p>
        </div>

        <div className="p-3 sm:p-4">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 text-xs uppercase tracking-wider border-b border-slate-700/50">
                  <th className="text-left pb-2.5 pl-2 font-medium">Rank</th>
                  <th className="text-left pb-2.5 font-medium">Source</th>
                  <th className="text-center pb-2.5 font-medium">MAE</th>
                  <th className="text-center pb-2.5 font-medium">Games</th>
                </tr>
              </thead>
              <tbody>
                {allByOuMae.map((b, i) => {
                  const isAlgo = b.book === 'AlgoPicks';
                  return (
                    <tr key={b.book} className={`border-b border-slate-700/30 ${i % 2 === 0 ? 'bg-slate-700/10' : ''}`}>
                      <td className="py-2.5 pl-2">
                        <RankBadge rank={i + 1} />
                      </td>
                      <td className="py-2.5">
                        <span className={`font-medium ${isAlgo ? 'text-orange-400' : 'text-slate-200'}`}>
                          {displayName(b.book)}
                        </span>
                        {i === 0 && <span className="text-orange-400 ml-1">&#9733;</span>}
                      </td>
                      <td className="py-2.5 text-center">
                        <span className="text-white font-semibold tabular-nums">{b.ou_mae!.toFixed(2)} pts</span>
                      </td>
                      <td className="py-2.5 text-center text-slate-400 tabular-nums">
                        {b.ou_games}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* AlgoPicks O/U Accuracy vs Each Book */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-base font-semibold text-white">AlgoPicks O/U vs Sportsbook Lines</h3>
          <p className="text-xs text-slate-500 mt-0.5">Accuracy of AlgoPicks total prediction against each book's O/U line</p>
        </div>

        <div className="p-3 sm:p-4">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 text-xs uppercase tracking-wider border-b border-slate-700/50">
                  <th className="text-left pb-2.5 pl-2 font-medium">vs Line</th>
                  <th className="text-center pb-2.5 font-medium">Record</th>
                  <th className="text-center pb-2.5 font-medium">Accuracy</th>
                  <th className="text-center pb-2.5 font-medium">Over</th>
                  <th className="text-center pb-2.5 font-medium">Under</th>
                </tr>
              </thead>
              <tbody>
                {booksByApOuAcc.map((b, i) => {
                  const acc = b.ap_ou_acc ?? 0;
                  return (
                    <tr key={b.book} className={`border-b border-slate-700/30 ${i % 2 === 0 ? 'bg-slate-700/10' : ''}`}>
                      <td className="py-2.5 pl-2 font-medium text-slate-200">{displayName(b.book)}</td>
                      <td className="py-2.5 text-center text-slate-400 tabular-nums">
                        {b.ap_ou_right}/{b.ap_ou_total}
                      </td>
                      <td className="py-2.5 text-center">
                        <span className={`font-semibold tabular-nums ${acc > 50 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {acc.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5 text-center tabular-nums">
                        <span className={`${(b.ap_over_acc ?? 0) > 50 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {b.ap_over_acc !== null ? `${b.ap_over_acc.toFixed(1)}%` : '-'}
                        </span>
                      </td>
                      <td className="py-2.5 text-center tabular-nums">
                        <span className={`${(b.ap_under_acc ?? 0) > 50 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {b.ap_under_acc !== null ? `${b.ap_under_acc.toFixed(1)}%` : '-'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
