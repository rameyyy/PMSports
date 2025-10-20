import { useNavigate } from 'react-router-dom';

export default function HomePage() {
  const navigate = useNavigate();

  const sports = [
    {
      name: "MMA",
      available: true,
      logo: "/logo/ufc-logo.png",
      gradient: "from-white to-blue",
      path: "/ufc"
    },
    {
      name: "Basketball",
      available: false,
      logo: "/logo/ncaa-logo.png",
      gradient: "from-white to-blue",
      path: "/cbb"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navbar */}
      <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <div className="px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <span className="text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-slate-300 hover:text-white transition-colors">
                About
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-20">
          <p className="text-2xl md:text-3xl text-slate-300 max-w-4xl mx-auto mb-4 font-medium">
            Machine Learning Models Predicting Tomorrow's Winners Today
          </p>
          <p className="text-base md:text-lg text-slate-500 max-w-3xl mx-auto">
            Built by a CS Undergraduate student at Auburn University
          </p>
        </div>

        {/* Sport Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {sports.map((sport) => (
            <button
              key={sport.name}
              disabled={!sport.available}
              onClick={() => {
                if (sport.available) {
                  navigate(sport.path);
                }
              }}
              className="group relative overflow-hidden rounded-2xl p-12 transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer"
            >
              {/* Gradient Background */}
              <div className={`absolute inset-0 bg-gradient-to-br ${sport.gradient} opacity-90 group-hover:opacity-100 transition-opacity`}></div>
              
              {/* Content */}
              <div className="relative z-10 flex flex-col items-center text-center">
                <img src={sport.logo} alt={sport.name} className="h-20 mb-8 object-contain" />
                <h2 className="text-4xl font-bold text-white mb-8">
                  {sport.name}
                </h2>
                
                {sport.available ? (
                  <div className="flex items-center text-white font-semibold text-lg">
                    View Predictions
                    <svg className="w-6 h-6 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                ) : (
                  <div className="inline-block px-5 py-2 bg-white/20 rounded-full text-white text-base font-semibold">
                    Coming Soon
                  </div>
                )}
              </div>

              {/* Shine effect on hover */}
              {sport.available && (
                <div className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent transform -skew-x-12 translate-x-full group-hover:translate-x-[-200%] transition-transform duration-1000"></div>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}