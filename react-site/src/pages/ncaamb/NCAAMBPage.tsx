import { useState, useEffect } from 'react';
import TabNav from './components/TabNav';
import DateSelector from './components/DateSelector';
import DayStats from './components/DayStats';
import GamesTable from './components/GamesTable';
import ModelPerformance from './components/ModelPerformance';
import { fetchGamesByDate, type Game } from '../../api/ncaamb';

const TABS = [
  { id: 'games', label: 'Games' },
  { id: 'performance', label: 'Model Performance' },
  { id: 'leaderboard', label: 'Leaderboard' },
];

function getDefaultDate() {
  const now = new Date();
  if (now.getHours() < 8) {
    now.setDate(now.getDate() - 1);
  }
  return now;
}

export default function NCAAMBPage() {
  const [activeTab, setActiveTab] = useState('games');
  const [selectedDate, setSelectedDate] = useState(getDefaultDate);
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);
  const [rankFilter, setRankFilter] = useState<'all' | 'top25' | 'top50'>('top25');
  const [confFilter, setConfFilter] = useState<string>('all');

  useEffect(() => {
    const dateStr = `${selectedDate.getFullYear()}-${String(selectedDate.getMonth() + 1).padStart(2, '0')}-${String(selectedDate.getDate()).padStart(2, '0')}`;
    setLoading(true);
    fetchGamesByDate(dateStr)
      .then((data) => setGames(data.games))
      .catch(() => setGames([]))
      .finally(() => setLoading(false));
  }, [selectedDate]);

  const isToday = selectedDate.toDateString() === new Date().toDateString();

  // Filter games
  const filteredGames = games.filter(g => {
    if (rankFilter === 'top25' && !(g.team_1_rank !== null && g.team_1_rank <= 25 || g.team_2_rank !== null && g.team_2_rank <= 25)) return false;
    if (rankFilter === 'top50' && !(g.team_1_rank !== null && g.team_1_rank <= 50 || g.team_2_rank !== null && g.team_2_rank <= 50)) return false;
    if (confFilter !== 'all' && g.team_1_conference !== confFilter && g.team_2_conference !== confFilter) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navbar */}
      <nav className="border-b border-slate-700 bg-slate-900 sticky top-0 z-50">
        <div className="px-4 sm:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => window.location.href = '/'}
              className="flex items-center hover:opacity-80 transition-opacity"
            >
              <span className="text-lg sm:text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </button>
            <div className="flex items-center space-x-2">
              <img src="/logo/ncaa-logo.png" alt="NCAA" className="h-6 sm:h-8 object-contain" />
              <span className="text-white font-semibold text-sm sm:text-base">Men's Basketball</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Tab Navigation */}
      <TabNav tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content */}
      <div className="max-w-[90rem] mx-auto px-2 sm:px-4 lg:px-8 py-4 sm:py-6">
        {/* Games Tab */}
        {activeTab === 'games' && (
          <>
            <DateSelector selectedDate={selectedDate} onDateChange={setSelectedDate} gameCount={filteredGames.length} totalGameCount={games.length} />

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-2 mb-3">
              {['all', 'top25', 'top50'].map((val) => (
                <button
                  key={val}
                  onClick={() => setRankFilter(val as typeof rankFilter)}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    rankFilter === val
                      ? 'bg-orange-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {val === 'all' ? 'All' : val === 'top25' ? 'Top 25' : 'Top 50'}
                </button>
              ))}

              <select
                value={confFilter}
                onChange={(e) => setConfFilter(e.target.value)}
                className="px-3 py-1 rounded text-sm font-medium bg-slate-700 text-slate-300 border border-slate-600 hover:bg-slate-600 transition-colors"
              >
                <option value="all">All Conferences</option>
                {[
                  { val: 'SEC', label: 'SEC' },
                  { val: 'B10', label: 'Big Ten' },
                  { val: 'ACC', label: 'ACC' },
                  { val: 'B12', label: 'Big 12' },
                  { val: 'BE', label: 'Big East' },
                  { val: 'WCC', label: 'WCC' },
                  { val: 'MWC', label: 'Mountain West' },
                  { val: 'A10', label: 'Atlantic 10' },
                  { val: 'Amer', label: 'American' },
                  { val: 'Ivy', label: 'Ivy League' },
                ].map((c) => (
                  <option key={c.val} value={c.val}>{c.label}</option>
                ))}
              </select>
            </div>

            {!isToday && <DayStats games={filteredGames} />}
            <GamesTable games={filteredGames} loading={loading} isToday={isToday} />
          </>
        )}

        {/* Model Performance Tab */}
        {activeTab === 'performance' && <ModelPerformance />}

        {/* Leaderboard Tab */}
        {activeTab === 'leaderboard' && (
          <div className="text-center text-slate-400 py-12">
            <h2 className="text-xl font-semibold text-white mb-2">Leaderboard</h2>
            <p>Coming soon...</p>
          </div>
        )}
      </div>
    </div>
  );
}
