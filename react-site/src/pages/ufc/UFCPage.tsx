import { useState, useEffect } from 'react';

type Fight = {
  fight_id: string;
  event_id: string;
  fighter1_id: string;
  fighter2_id: string;
  winner_id?: string;
  loser_id?: string;
  fighter1_name: string;
  fighter2_name: string;
  fight_date: string;
  fight_link: string;
  method?: string;
  fight_format: string;
  fight_type?: string;
  referee?: string;
  end_time?: string;
  weight_class: string;
  fighter1_nickname?: string;
  fighter1_img?: string;
  fighter2_nickname?: string;
  fighter2_img?: string;
  logistic_pred?: number;
  logistic_f1_prob?: number;
  logistic_correct?: number;
  xgboost_pred?: number;
  xgboost_f1_prob?: number;
  xgboost_correct?: number;
  gradient_pred?: number;
  gradient_f1_prob?: number;
  gradient_correct?: number;
  homemade_pred?: number;
  homemade_f1_prob?: number;
  homemade_correct?: number;
  prediction_confidence?: number;
  predicted_winner?: number;
};

type Event = {
  event_id: string;
  event_url: string;
  title: string;
  event_datestr: string;
  location: string;
  date: string;
};

// Helper function to capitalize names including hyphens
function capitalizeName(name: string): string {
  return name
    .split(' ')
    .map(word => {
      // Handle hyphenated names like "al-dew" -> "Al-Dew"
      return word
        .split('-')
        .map(part => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
        .join('-');
    })
    .join(' ');
}

// Helper function to format date
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const options: Intl.DateTimeFormatOptions = { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric' 
  };
  return date.toLocaleDateString('en-US', options);
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
  const [selectedModel, setSelectedModel] = useState<{ [key: string]: string }>({});

  // Fetch events when component mounts
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

  // Fetch fights when an event is expanded
  const handleEventClick = async (eventId: string) => {
    if (expandedEvent === eventId) {
      setExpandedEvent(null);
      return;
    }

    setExpandedEvent(eventId);

    // Only fetch if we haven't already
    if (!eventFights[eventId]) {
      try {
        const response = await fetch(`http://localhost:5000/api/ufc/events/${eventId}/fights`);
        const fights = await response.json();
        setEventFights(prev => ({ ...prev, [eventId]: fights }));
        
        // Initialize selected model for each fight as "ensemble_avg"
        const initialModels: { [key: string]: string } = {};
        fights.forEach((fight: Fight) => {
          initialModels[fight.fight_id] = 'ensemble_avg';
        });
        setSelectedModel(prev => ({ ...prev, ...initialModels }));
      } catch (error) {
        console.error('Error fetching fights:', error);
      }
    }
  };

  const currentEvents = events[activeTab as keyof typeof events];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading events...</div>
      </div>
    );
  }

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
                {/* Event Header */}
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

                {/* Expanded Fights */}
                {expandedEvent === event.event_id && (
                  <div className="border-t border-slate-700">
                    {!eventFights[event.event_id] ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        Loading fights...
                      </div>
                    ) : eventFights[event.event_id].length === 0 ? (
                      <div className="px-6 py-8 text-center text-slate-400">
                        No fights available for this event
                      </div>
                    ) : (
                      eventFights[event.event_id].map((fight) => {
                        const isPastFight = fight.winner_id !== null;
                        
                        // Get current selected model for this fight
                        const currentModel = selectedModel[fight.fight_id] || 'ensemble_avg';
                        
                        // Get probabilities based on selected model
                        let fighter1Prob = 0.5;
                        let predictedWinner = 0;
                        
                        if (currentModel === 'logistic') {
                          fighter1Prob = fight.logistic_f1_prob || 0.5;
                          predictedWinner = fight.logistic_pred || 0;
                        } else if (currentModel === 'xgboost') {
                          fighter1Prob = fight.xgboost_f1_prob || 0.5;
                          predictedWinner = fight.xgboost_pred || 0;
                        } else if (currentModel === 'gradient') {
                          fighter1Prob = fight.gradient_f1_prob || 0.5;
                          predictedWinner = fight.gradient_pred || 0;
                        } else if (currentModel === 'homemade') {
                          fighter1Prob = fight.homemade_f1_prob || 0.5;
                          predictedWinner = fight.homemade_pred || 0;
                        } else if (currentModel === 'ensemble_weighted') {
                          fighter1Prob = fight.prediction_confidence || 0.5;
                          predictedWinner = fight.predicted_winner || 0;
                        } else { // ensemble_avg
                          // Calculate average of all models
                          const probs = [
                            fight.logistic_f1_prob,
                            fight.xgboost_f1_prob,
                            fight.gradient_f1_prob,
                            fight.homemade_f1_prob
                          ].filter(p => p !== null && p !== undefined) as number[];
                          
                          fighter1Prob = probs.length > 0 ? probs.reduce((a, b) => a + b, 0) / probs.length : 0.5;
                          predictedWinner = fighter1Prob >= 0.5 ? 1 : 0;
                        }
                        
                        const fighter2Prob = 1 - fighter1Prob;
                        
                        // Determine predicted winner based on prediction (1 = fighter1, 0 = fighter2)
                        const predictedWinnerId = predictedWinner === 1 ? fight.fighter1_id : fight.fighter2_id;
                        const predictedWinnerName = predictedWinner === 1 ? fight.fighter1_name : fight.fighter2_name;
                        
                        // Calculate confidence (higher probability)
                        const confidence = Math.max(fighter1Prob, fighter2Prob) * 100;
                        
                        // Check if prediction was correct
                        const predictionCorrect = (predictedWinner === 1 && fight.winner_id === fight.fighter1_id) || 
                                                 (predictedWinner === 0 && fight.winner_id === fight.fighter2_id);

                        // Model display names
                        const models = [
                          { value: 'ensemble_avg', label: '⭐ Ensemble Avg', isDefault: true },
                          { value: 'ensemble_weighted', label: 'Ensemble Weighted Avg', isDefault: false },
                          { value: 'xgboost', label: 'XGBoost', isDefault: false },
                          { value: 'gradient', label: 'Gradient Boosting', isDefault: false },
                          { value: 'logistic', label: 'Logistic Regression', isDefault: false }
                        ];

                        return (
                          <div
                            key={fight.fight_id}
                            className="px-6 py-4 border-b border-slate-700/50 last:border-0"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-2">
                                <span className="text-xs font-semibold text-orange-500 uppercase tracking-wide">
                                  {fight.weight_class}
                                </span>
                                {fight.fight_type === 'title' && (
                                  <span className="text-xs font-semibold px-2 py-1 rounded bg-yellow-500/20 text-yellow-400">
                                    TITLE FIGHT
                                  </span>
                                )}
                                {fight.fight_type === 'main' && (
                                  <span className="text-xs font-semibold px-2 py-1 rounded bg-blue-500/20 text-blue-400">
                                    MAIN EVENT
                                  </span>
                                )}
                              </div>
                              {isPastFight && (
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
                              {/* Fighter 1 */}
                              <div className="flex flex-col items-end">
                                {fight.fighter1_img && (
                                  <img 
                                    src={fight.fighter1_img} 
                                    alt={fight.fighter1_name}
                                    className="w-16 h-16 rounded-full object-cover mb-2 border-2 border-slate-600"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                    }}
                                  />
                                )}
                                <p className="text-white font-semibold text-lg text-right">{capitalizeName(fight.fighter1_name)}</p>
                                {fight.fighter1_nickname && (
                                  <p className="text-slate-400 text-sm italic text-right">{capitalizeName(fight.fighter1_nickname)}</p>
                                )}
                              </div>
                              
                              {/* VS */}
                              <div className="text-center">
                                <span className="text-slate-500 font-bold text-xl">VS</span>
                              </div>
                              
                              {/* Fighter 2 */}
                              <div className="flex flex-col items-start">
                                {fight.fighter2_img && (
                                  <img 
                                    src={fight.fighter2_img} 
                                    alt={fight.fighter2_name}
                                    className="w-16 h-16 rounded-full object-cover mb-2 border-2 border-slate-600"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                    }}
                                  />
                                )}
                                <p className="text-white font-semibold text-lg text-left">{capitalizeName(fight.fighter2_name)}</p>
                                {fight.fighter2_nickname && (
                                  <p className="text-slate-400 text-sm italic text-left">{capitalizeName(fight.fighter2_nickname)}</p>
                                )}
                              </div>
                            </div>

                            <div className="flex items-center justify-between text-sm mb-3">
                              <div>
                                <span className="text-slate-400">Prediction: </span>
                                <span className="text-orange-400 font-semibold">
                                  {capitalizeName(predictedWinnerName)}
                                </span>
                                <span className="text-slate-500 ml-2">({confidence.toFixed(1)}% confidence)</span>
                              </div>
                            </div>

                            {/* Model Selector */}
                            <div className="mb-3">
                              <div className="flex flex-wrap gap-2">
                                {models.map((model) => (
                                  <button
                                    key={model.value}
                                    onClick={() => setSelectedModel(prev => ({
                                      ...prev,
                                      [fight.fight_id]: model.value
                                    }))}
                                    className={`text-xs px-3 py-1 rounded-full transition-colors ${
                                      currentModel === model.value
                                        ? 'bg-orange-500 text-white font-semibold'
                                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                                    }`}
                                  >
                                    {model.label}
                                  </button>
                                ))}
                              </div>
                            </div>

                            {/* Model Prediction Bar */}
                            <div className="mt-4">
                              <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden">
                                <div 
                                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-orange-500 to-orange-600 transition-all duration-500"
                                  style={{ width: `${fighter1Prob * 100}%` }}
                                ></div>
                                <div 
                                  className="absolute top-0 right-0 h-full bg-gradient-to-r from-slate-600 to-slate-700 transition-all duration-500"
                                  style={{ width: `${fighter2Prob * 100}%` }}
                                ></div>
                                <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-semibold text-white">
                                  <span>{capitalizeName(fight.fighter1_name.split(' ').pop() || '')} {(fighter1Prob * 100).toFixed(1)}%</span>
                                  <span>{capitalizeName(fight.fighter2_name.split(' ').pop() || '')} {(fighter2Prob * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            </div>

                            {isPastFight && fight.method && (
                              <div className="mt-3 text-sm text-slate-400">
                                Result: <span className="text-slate-300">{fight.method}</span>
                                {fight.end_time && <span> at {fight.end_time}</span>}
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