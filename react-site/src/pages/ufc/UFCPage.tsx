import { useState } from 'react';

export default function UFCPage() {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);

  // Mock data - replace with API calls later
  const events = {
    upcoming: [
      {
        id: '1',
        name: 'UFC 310: Pantoja vs Asakura',
        date: 'Dec 7, 2024',
        location: 'T-Mobile Arena, Las Vegas',
        fights: [
          {
            fighter1: 'Alexandre Pantoja',
            fighter2: 'Kai Asakura',
            weightClass: 'Flyweight Title',
            prediction: 'Pantoja',
            confidence: 68,
            odds1: -180,
            odds2: +150,
            vig: 4.5
          },
          {
            fighter1: 'Shavkat Rakhmonov',
            fighter2: 'Ian Machado Garry',
            weightClass: 'Welterweight',
            prediction: 'Rakhmonov',
            confidence: 72,
            odds1: -240,
            odds2: +195,
            vig: 3.8
          }
        ]
      },
      {
        id: '2',
        name: 'UFC Fight Night: Moreno vs Albazi',
        date: 'Nov 2, 2024',
        location: 'Rogers Place, Edmonton',
        fights: [
          {
            fighter1: 'Brandon Moreno',
            fighter2: 'Amir Albazi',
            weightClass: 'Flyweight',
            prediction: 'Moreno',
            confidence: 61,
            odds1: -160,
            odds2: +135,
            vig: 4.2
          }
        ]
      }
    ],
    past: [
      {
        id: '3',
        name: 'UFC 309: Jones vs Miocic',
        date: 'Nov 16, 2024',
        location: 'Madison Square Garden, New York',
        fights: [
          {
            fighter1: 'Jon Jones',
            fighter2: 'Stipe Miocic',
            weightClass: 'Heavyweight Title',
            prediction: 'Jones',
            confidence: 85,
            odds1: -650,
            odds2: +450,
            result: 'Jones',
            correct: true,
            vig: 5.1
          }
        ]
      }
    ]
  };

  const currentEvents = events[activeTab as keyof typeof events];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Top Navigation */}
      <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="px-8">
          <div className="flex items-center justify-between h-16">
            <button 
              onClick={() => window.location.href = '/'}
              className="flex items-center hover:opacity-80 transition-opacity"
            >
              <span className="text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </button>
            <div className="flex items-center">
              <img src="/logo/ufc-logo.png" alt="UFC" className="h-8 object-contain" />
            </div>
          </div>
        </div>
      </nav>

      {/* Tab Bar */}
      <div className="border-b border-slate-700 bg-slate-900/30 sticky top-16 z-40">
        <div className="px-8 flex items-center space-x-6">
          <button
            onClick={() => setActiveTab('upcoming')}
            className={`py-4 px-2 font-semibold transition-colors border-b-2 ${
              activeTab === 'upcoming'
                ? 'text-orange-500 border-orange-500'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Upcoming
          </button>
          <button
            onClick={() => setActiveTab('past')}
            className={`py-4 px-2 font-semibold transition-colors border-b-2 ${
              activeTab === 'past'
                ? 'text-orange-500 border-orange-500'
                : 'text-slate-400 border-transparent hover:text-slate-300'
            }`}
          >
            Past
          </button>
          <button
            onClick={() => alert('Model Results page coming soon!')}
            className="py-4 px-4 font-semibold text-slate-400 hover:text-slate-300 transition-colors ml-auto"
          >
            Model Results →
          </button>
        </div>
      </div>

      {/* Events List */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-4">
          {currentEvents.map((event) => (
            <div
              key={event.id}
              className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden"
            >
              {/* Event Header */}
              <button
                onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/80 transition-colors"
              >
                <div className="text-left">
                  <h3 className="text-xl font-bold text-white mb-1">{event.name}</h3>
                  <p className="text-slate-400 text-sm">
                    {event.date} • {event.location}
                  </p>
                </div>
                <svg
                  className={`w-6 h-6 text-slate-400 transition-transform ${
                    expandedEvent === event.id ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Expanded Fights */}
              {expandedEvent === event.id && (
                <div className="border-t border-slate-700">
                  {event.fights.map((fight, idx) => (
                    <div
                      key={idx}
                      className="px-6 py-4 border-b border-slate-700/50 last:border-0"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-semibold text-orange-500 uppercase tracking-wide">
                          {fight.weightClass}
                        </span>
                        {fight.result && (
                          <span
                            className={`text-xs font-semibold px-2 py-1 rounded ${
                              fight.correct
                                ? 'bg-green-500/20 text-green-400'
                                : 'bg-red-500/20 text-red-400'
                            }`}
                          >
                            {fight.correct ? '✓ Correct' : '✗ Incorrect'}
                          </span>
                        )}
                      </div>

                      <div className="grid grid-cols-3 gap-4 items-center mb-3">
                        <div className="text-right">
                          <p className="text-white font-semibold">{fight.fighter1}</p>
                          <p className="text-slate-400 text-sm">{fight.odds1 > 0 ? '+' : ''}{fight.odds1}</p>
                        </div>
                        <div className="text-center">
                          <span className="text-slate-500 font-bold">VS</span>
                        </div>
                        <div className="text-left">
                          <p className="text-white font-semibold">{fight.fighter2}</p>
                          <p className="text-slate-400 text-sm">{fight.odds2 > 0 ? '+' : ''}{fight.odds2}</p>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <div>
                          <span className="text-slate-400">Prediction: </span>
                          <span className="text-orange-400 font-semibold">{fight.prediction}</span>
                          <span className="text-slate-500 ml-2">({fight.confidence}% confidence)</span>
                        </div>
                        <div>
                          <span className="text-slate-400">Vig: </span>
                          <span className="text-slate-300">{fight.vig}%</span>
                        </div>
                      </div>

                      {/* Model Prediction Bar */}
                      <div className="mt-4">
                        <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                          <span>Model: Logistic Regression</span>
                        </div>
                        <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden">
                          <div 
                            className="absolute top-0 left-0 h-full bg-gradient-to-r from-orange-500 to-orange-600 transition-all duration-500"
                            style={{ width: `${fight.confidence}%` }}
                          ></div>
                          <div 
                            className="absolute top-0 right-0 h-full bg-gradient-to-r from-slate-600 to-slate-700 transition-all duration-500"
                            style={{ width: `${100 - fight.confidence}%` }}
                          ></div>
                          <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-semibold text-white">
                            <span>{fight.fighter1} {fight.confidence}%</span>
                            <span>{fight.fighter2} {100 - fight.confidence}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}