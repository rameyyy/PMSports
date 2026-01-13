import { useNavigate } from 'react-router-dom';

interface Sport {
  name: string;
  logo: string;
  path: string;
  available: boolean;
}

interface SportNavBarProps {
  sports: Sport[];
}

export default function SportNavBar({ sports }: SportNavBarProps) {
  const navigate = useNavigate();

  return (
    <div className="border-t border-slate-700/50 bg-slate-900/30">
      <div className="px-8 py-3 md:py-3.5">
        <div className="flex items-center gap-6 md:gap-10">
          {sports.map((sport) => (
            <button
              key={sport.name}
              onClick={() => sport.available && navigate(sport.path)}
              className={`flex items-center gap-2 md:gap-2.5 ${
                sport.available
                  ? 'text-white hover:text-orange-500 transition-colors'
                  : 'text-slate-300 cursor-default'
              }`}
              disabled={!sport.available}
            >
              <img src={sport.logo} alt={sport.name} className="h-5 w-5 md:h-6 md:w-6 object-contain" />
              <span className="text-sm md:text-base">{sport.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
