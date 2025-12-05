CREATE TABLE my_bankroll (
  date DATE NOT NULL PRIMARY KEY,
  bankroll DECIMAL(12, 2) NOT NULL,
  net_profit_loss DECIMAL(12, 2),
  roi DECIMAL(8, 4),

  bet_qty INT DEFAULT 0,
  ml_bets INT DEFAULT 0,
  ou_bets INT DEFAULT 0,
  spread_bets INT DEFAULT 0,

  total_wagered DECIMAL(12, 2) DEFAULT 0,
  ml_wagered DECIMAL(12, 2) DEFAULT 0,
  ou_wagered DECIMAL(12, 2) DEFAULT 0,
  spread_wagered DECIMAL(12, 2) DEFAULT 0,

  ml_net_profit_loss DECIMAL(12, 2),
  ou_net_profit_loss DECIMAL(12, 2),
  spread_net_profit_loss DECIMAL(12, 2),

  ml_roi DECIMAL(8, 4),
  ou_roi DECIMAL(8, 4),
  spread_roi DECIMAL(8, 4),

  ml_wins INT DEFAULT 0,
  ou_wins INT DEFAULT 0,
  spread_wins INT DEFAULT 0,

  cumulative_profit DECIMAL(12, 2),
  season INT,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);


CREATE TABLE bankroll (
  date DATE NOT NULL PRIMARY KEY,
  bankroll DECIMAL(12, 2) NOT NULL,
  net_profit_loss DECIMAL(12, 2),
  roi DECIMAL(8, 4),

  bet_qty INT DEFAULT 0,
  ml_bets INT DEFAULT 0,
  ou_bets INT DEFAULT 0,
  spread_bets INT DEFAULT 0,

  total_wagered DECIMAL(12, 2) DEFAULT 0,
  ml_wagered DECIMAL(12, 2) DEFAULT 0,
  ou_wagered DECIMAL(12, 2) DEFAULT 0,
  spread_wagered DECIMAL(12, 2) DEFAULT 0,

  ml_net_profit_loss DECIMAL(12, 2),
  ou_net_profit_loss DECIMAL(12, 2),
  spread_net_profit_loss DECIMAL(12, 2),

  ml_roi DECIMAL(8, 4),
  ou_roi DECIMAL(8, 4),
  spread_roi DECIMAL(8, 4),

  ml_wins INT DEFAULT 0,
  ou_wins INT DEFAULT 0,
  spread_wins INT DEFAULT 0,

  cumulative_profit DECIMAL(12, 2),
  season INT,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
