import { useState, useEffect } from 'react';

type Fight = {
  fight_id: string;
  event_id: string;
  fighter1_id: string;
  fighter2_id: string;
  fighter1_name: string;
  fighter2_name: string;
  fighter1_nickname: string | null;
  fighter2_nickname: string | null;
  fighter1_img_link: string | null;
  fighter2_img_link: string | null;
  algopick_model: string | null;
  algopick_prediction: number | null;
  algopick_probability: number | null;
  window_sample: number | null;
  correct: number | null;
  date: string;
  end_time: string | null;
  weight_class: string;
  win_method: string | null;
};

type Event = {
  event_id: string;
  event_url: string;
  title: string;
  event_datestr: string;
  location: string;
  date: string;
};

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const options: Intl.DateTimeFormatOptions = { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric' 
  };
  return date.toLocaleDateString('en-US', options);
}

function getModelDisplayName(modelKey: string): string {
  const modelMap: { [key: string]: string } = {
    'logistic': 'Logistic Regression',
    'xgboost': 'XGBoost',
    'gradient': 'Gradient Boosting',
    'homemade': 'Homemade Model',
    'ensemble_weightedvote': 'Ensemble Weighted Vote',
    'ensemble_avgprob': 'Ensemble Avg Prob',
    'ensemble_weightedavgprob': 'Ensemble Weighted Avg Prob'
  };
  return modelMap[modelKey] || modelKey;
}

export default function UFCPage() {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [events, setEvents] = useState<{ upcoming: Event[]; past: Event[] }>({ 
    upcoming: [], 
    past: [] 
  });
  const [eventFights, setEventFights] = useState<{ [key: string]: Fight[] }>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const upcomingRes = await fetch('http://localhost:5000/api/ufc/events/upcoming');
        const pastRes = await fetch('http://localhost:5000/api/ufc/events/past');
        
        const upcomingData = await upcomingRes.json();
        const pastData = await pastRes.json();
        
        setEvents({
          upcoming: upcomingData,
          past: pastData
        });
        setLoading(false);
      } catch (error) {
        console.error('Error fetching events:', error);
        setLoading(false);
      }
    };

    fetchEvents();
  }, []);

  const handleEventClick = async (eventId: string) => {
    if (expandedEvent === eventId) {
      setExpandedEvent(null);
      return;
    }

    setExpandedEvent(eventId);

    if (!eventFights[eventId]) {
      try {
        console.log('Fetching fights for event:', eventId);
        const response = await fetch(`http://localhost:5000/api/ufc/events/${eventId}/fights`);
        const fights = await response.json();
        console.log('Received fights:', fights);
        console.log('First fight:', fights[0]);
        
        // Sort fights: title first, then main, then rest
        const sortedFights = fights.sort((a: Fight, b: Fight) => {
          if (a.fight_type === 'title' && b.fight_type !== 'title') return -1;
          if (a.fight_type !== 'title' && b.fight_type === 'title') return 1;
          if (a.fight_type === 'main' && b.fight_type !== 'main') return -1;
          if (a.fight_type !== 'main' && b.fight_type === 'main') return 1;
          return 0;
        });
        
        setEventFights(prev => ({ ...prev, [eventId]: sortedFights }));
        console.log('Successfully set event fights');
      } catch (error) {
        console.error('Error fetching fights:', error);
      }
    }
  };

  const currentEvents = events[activeTab as keyof typeof events];
  const isPastTab = activeTab === 'past';

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading events...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
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
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentEvents.length === 0 ? (
          <div className="text-center text-slate-400 py-12">
            No {activeTab} events found
          </div>
        ) : (
          <div className="space-y-4">
            {currentEvents.map((event) => (
              <div
                key={event.event_id}
                className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden"
              >
                <button
                  onClick={() => handleEventClick(event.event_id)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-800/80 transition-colors"
                >
                  <div className="text-left">
                    <h3 className="text-xl font-bold text-white mb-1">{event.title}</h3>
                    <p className="text-slate-400 text-sm">{formatDate(event.date)}</p>
                    <p className="text-slate-500 text-xs">{event.location}</p>
                  </div>
                  <svg
                    className={`w-6 h-6 text-slate-400 transition-transform ${
                      expandedEvent === event.event_id ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {expandedEvent === event.event_id && (
                  <div className="border-t border-slate-700">
                    {!eventFights[event.event_id] ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        Loading fights...
                      </div>
                    ) : eventFights[event.event_id].length === 0 ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        No fights available for this event
                        <div className="text-xs mt-2">Debug: {JSON.stringify(eventFights[event.event_id])}</div>
                      </div>
                    ) : (
                      eventFights[event.event_id].map((fight) => {
                        console.log('Rendering fight:', fight);
                        const hasPrediction = fight.algopick_prediction !== null && 
                                            fight.algopick_prediction !== undefined &&
                                            fight.algopick_probability !== null;
                        
                        if (!hasPrediction) {
                          return (
                            <div key={fight.fight_id} className="px-6 py-4 border-b border-slate-700/50 last:border-0">
                              <div className="grid grid-cols-3 gap-4 items-center mb-2">
                                <p className="text-white font-semibold text-right">{fight.fighter1_name}</p>
                                <span className="text-slate-500 font-bold text-center">VS</span>
                                <p className="text-white font-semibold text-left">{fight.fighter2_name}</p>
                              </div>
                              <div className="text-center text-slate-500 text-sm mt-2">
                                No prediction available
                              </div>
                            </div>
                          );
                        }

                        const predictedWinnerName = fight.algopick_prediction === 0 
                          ? fight.fighter1_name 
                          : fight.fighter2_name;
                        
                        // Convert probability from string to number
                        const confidence = parseFloat(fight.algopick_probability);
                        const fighter1Prob = fight.algopick_prediction === 0 ? confidence : (100 - confidence);
                        const fighter2Prob = 100 - fighter1Prob;
                        
                        const showResult = isPastTab && fight.correct !== null;
                        const predictionCorrect = fight.correct === 1;

                        return (
                          <div key={fight.fight_id} className="px-6 py-4 border-b border-slate-700/50 last:border-0">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-xs font-semibold text-orange-500 uppercase tracking-wide">
                                  {fight.weight_class}
                                </span>
                                {fight.fight_type === 'title' && (
                                  <span className="text-xs font-semibold px-3 py-1 rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                                    🏆 TITLE FIGHT
                                  </span>
                                )}
                                {fight.fight_type === 'main' && (
                                  <span className="text-xs font-semibold px-3 py-1 rounded bg-red-500/20 text-red-400 border border-red-500/30">
                                    ⭐ MAIN EVENT
                                  </span>
                                )}
                              </div>
                              {showResult && (
                                <span
                                  className={`text-xs font-semibold px-2 py-1 rounded ${
                                    predictionCorrect
                                      ? 'bg-green-500/20 text-green-400'
                                      : 'bg-red-500/20 text-red-400'
                                  }`}
                                >
                                  {predictionCorrect ? '✓ Correct' : '✗ Incorrect'}
                                </span>
                              )}
                            </div>

                            <div className="grid grid-cols-3 gap-4 items-center mb-4">
                              <div className="flex flex-col items-end">
                                {fight.fighter1_img_link && (
                                  <img 
                                    src={fight.fighter1_img_link} 
                                    alt={fight.fighter1_name}
                                    className={`w-16 h-16 rounded-full object-cover mb-2 ${
                                      fight.algopick_prediction === 0 
                                        ? 'border-4 border-purple-500 shadow-lg shadow-purple-500/50' 
                                        : 'border-2 border-slate-600'
                                    }`}
                                    onError={(e) => e.currentTarget.style.display = 'none'}
                                  />
                                )}
                                <p className={`font-semibold text-lg text-right ${
                                  fight.algopick_prediction === 0 ? 'text-purple-400' : 'text-white'
                                }`}>
                                  {fight.fighter1_name}
                                </p>
                                {fight.fighter1_nickname && (
                                  <p className="text-slate-400 text-sm italic text-right">
                                    "{fight.fighter1_nickname}"
                                  </p>
                                )}
                              </div>
                              
                              <div className="text-center">
                                <span className="text-slate-500 font-bold text-xl">VS</span>
                              </div>
                              
                              <div className="flex flex-col items-start">
                                {fight.fighter2_img_link && (
                                  <img 
                                    src={fight.fighter2_img_link} 
                                    alt={fight.fighter2_name}
                                    className={`w-16 h-16 rounded-full object-cover mb-2 ${
                                      fight.algopick_prediction === 1 
                                        ? 'border-4 border-cyan-500 shadow-lg shadow-cyan-500/50' 
                                        : 'border-2 border-slate-600'
                                    }`}
                                    onError={(e) => e.currentTarget.style.display = 'none'}
                                  />
                                )}
                                <p className={`font-semibold text-lg text-left ${
                                  fight.algopick_prediction === 1 ? 'text-cyan-400' : 'text-white'
                                }`}>
                                  {fight.fighter2_name}
                                </p>
                                {fight.fighter2_nickname && (
                                  <p className="text-slate-400 text-sm italic text-left">
                                    "{fight.fighter2_nickname}"
                                  </p>
                                )}
                              </div>
                            </div>

                            <div className="text-sm mb-3">
                              <span className="text-slate-400 font-bold">AlgoPick: </span>
                              <span className={`font-bold text-lg ${
                                fight.algopick_prediction === 0 ? 'text-purple-400' : 'text-cyan-400'
                              }`}>
                                {predictedWinnerName}
                              </span>
                              <span className="text-slate-500 ml-2 text-base">
                                ({confidence.toFixed(1)}%)
                              </span>
                            </div>

                            <div className="mt-4">
                              <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden">
                                <div 
                                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-purple-500 to-purple-600 transition-all"
                                  style={{ width: `${fighter1Prob}%` }}
                                ></div>
                                <div 
                                  className="absolute top-0 right-0 h-full bg-gradient-to-r from-cyan-500 to-cyan-600 transition-all"
                                  style={{ width: `${fighter2Prob}%` }}
                                ></div>
                                <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-semibold text-white">
                                  <span>{fighter1Prob.toFixed(1)}%</span>
                                  <span>{fighter2Prob.toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>

                            {showResult && fight.win_method && (
                              <div className="mt-3 text-sm">
                                <span className="text-slate-400">Finish: </span>
                                <span className="text-slate-200 font-medium">{fight.win_method}</span>
                                {fight.end_time && <span className="text-slate-400"> ({fight.end_time})</span>}
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}