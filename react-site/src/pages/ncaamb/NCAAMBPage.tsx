import { useState } from 'react';
import TabNav from './components/TabNav';
import DateSelector from './components/DateSelector';
import DayStats from './components/DayStats';
import GamesTable from './components/GamesTable';

const TABS = [
  { id: 'games', label: 'Games' },
  { id: 'performance', label: 'Model Performance' },
  { id: 'leaderboard', label: 'Leaderboard' },
];

export default function NCAAMBPage() {
  const [activeTab, setActiveTab] = useState('games');
  const [selectedDate, setSelectedDate] = useState(new Date());

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
            <DateSelector selectedDate={selectedDate} onDateChange={setSelectedDate} gameCount={5} />
            {selectedDate < new Date(new Date().setHours(0, 0, 0, 0)) && (
              <DayStats
                modelRecord={{ wins: 12, losses: 6 }}
                vegasRecord={{ wins: 10, losses: 8 }}
                modelMAE={8.2}
                vegasMAE={9.7}
                avgConfidence={64.3}
                avgVig={4.5}
                edgeAfterVig={6.6}
              />
            )}
            <GamesTable />
          </>
        )}

        {/* Model Performance Tab */}
        {activeTab === 'performance' && (
          <div className="text-center text-slate-400 py-12">
            <h2 className="text-xl font-semibold text-white mb-2">Model Performance</h2>
            <p>Coming soon...</p>
          </div>
        )}

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
