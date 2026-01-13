import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
      <div className="px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <span className="text-2xl font-bold text-white">
              Algo<span className="text-orange-500">Picks</span>
            </span>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate('/about')}
              className="text-slate-300 hover:text-white transition-colors"
            >
              About
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
