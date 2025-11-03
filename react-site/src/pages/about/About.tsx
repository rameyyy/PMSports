import { useNavigate } from 'react-router-dom';

export default function About() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navbar */}
      <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="px-4 sm:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => navigate('/')}
              className="flex items-center hover:opacity-80 transition-opacity"
            >
              <span className="text-xl sm:text-2xl font-bold text-white">
                Algo<span className="text-orange-500">Picks</span>
              </span>
            </button>
            <button
              onClick={() => navigate('/')}
              className="text-slate-300 hover:text-white transition-colors text-sm sm:text-base"
            >
              Back
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        {/* About Section */}
        <div className="mb-16 sm:mb-20">
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-6">About</h2>
          <p className="text-slate-300 text-base sm:text-lg mb-6 leading-relaxed">
            AlgoPicks uses machine learning models trained on historical sports data to predict game outcomes. Multiple models work together to generate probability-based predictions that help identify valuable betting opportunities.
          </p>
          <div className="space-y-6">
            <div>
              <h3 className="text-white font-semibold mb-3 text-base sm:text-lg">What Sports Are Covered?</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <img src="/logo/ufc-logo.png" alt="UFC" className="h-6 sm:h-8 object-contain flex-shrink-0" />
                  <p className="text-slate-300 text-sm sm:text-base">Mixed Martial Arts predictions</p>
                </div>
                <div className="flex items-center gap-3">
                  <img src="/logo/ncaa-logo.png" alt="NCAA" className="h-6 sm:h-8 object-contain flex-shrink-0" />
                  <p className="text-slate-300 text-sm sm:text-base">Men's Basketball predictions</p>
                </div>
                <p className="text-slate-400 text-sm mt-3">More sports coming soon...</p>
              </div>
            </div>

            <div>
              <h3 className="text-white font-semibold mb-3 text-base sm:text-lg">Model Analytics</h3>
              <div className="space-y-4">
                <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-3 mb-2">
                    <img src="/logo/ufc-logo.png" alt="UFC" className="h-5 sm:h-6 object-contain flex-shrink-0" />
                    <h4 className="text-white font-semibold text-sm sm:text-base">UFC Model</h4>
                  </div>
                  <p className="text-slate-300 text-sm">Trained on <span className="font-semibold text-orange-400">2,193 fights</span> with a test accuracy of <span className="font-semibold text-orange-400">74.7%</span></p>
                </div>
                <div className="p-4 bg-slate-700/30 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-3 mb-2">
                    <img src="/logo/ncaa-logo.png" alt="NCAA" className="h-5 sm:h-6 object-contain flex-shrink-0" />
                    <h4 className="text-white font-semibold text-sm sm:text-base">NCAAB Model</h4>
                  </div>
                  <p className="text-slate-300 text-sm">Advanced model using historic betting data and trained on <span className="font-semibold text-orange-400">45,000+ games</span>. Currently under development.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center pt-8 sm:pt-12 border-t border-slate-700">
          <button
            onClick={() => navigate('/')}
            className="inline-flex items-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-bold rounded-lg transition-all transform hover:scale-105 text-sm sm:text-base"
          >
            Home
            <svg className="w-5 h-5 sm:w-6 sm:h-6 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
