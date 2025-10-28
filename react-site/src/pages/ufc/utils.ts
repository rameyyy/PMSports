export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const options: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC'
  };
  return date.toLocaleDateString('en-US', options);
}

export function formatDateLong(dateString: string): string {
  const date = new Date(dateString);
  const options: Intl.DateTimeFormatOptions = {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC'
  };
  return date.toLocaleDateString('en-US', options);
}

export function formatDateShort(dateString: string): string {
  const match = dateString.match(/(\d{1,2})\s+(\w{3})\s+(\d{4})/);
  if (!match) return dateString;

  const [, day, monthStr, year] = match;
  const months: { [key: string]: string } = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
  };

  const month = months[monthStr];
  return `${month}-${day.padStart(2, '0')}-${year}`;
}

export function formatOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`;
}

export function getBookmakerDisplayName(key: string): string {
  const names: { [key: string]: string } = {
    'bovada': 'Bovada',
    'fanduel': 'FanDuel',
    'draftkings': 'DraftKings',
    'betmgm': 'BetMGM',
    'betonlineag': 'BetOnline',
    'betus': 'BetUS',
    'betrivers': 'BetRivers'
  };
  return names[key] || key;
}

export function calculateROI(net_profit: number, total_staked: number): number {
  if (total_staked === null || total_staked === 0 || net_profit === null) {
    return 0;
  }
  return (net_profit / total_staked) * 100;
}

export function formatNetProfit(amount: number): string {
  if (amount >= 0) {
    return `+$${amount.toFixed(2)}`;
  } else {
    return `-$${Math.abs(amount).toFixed(2)}`;
  }
}
