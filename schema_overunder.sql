CREATE TABLE overunder (
  id INT AUTO_INCREMENT PRIMARY KEY,
  game_id VARCHAR(255) NOT NULL,
  team_1 VARCHAR(100) NOT NULL,
  team_2 VARCHAR(100) NOT NULL,

  over_point DECIMAL(6, 2),

  xgb_pred DECIMAL(6, 2),
  lgb_pred DECIMAL(6, 2),
  cb_pred DECIMAL(6, 2),
  ensemble_pred DECIMAL(6, 2),
  ensemble_confidence DECIMAL(5, 4),

  good_bets_confidence DECIMAL(5, 4),
  difference DECIMAL(6, 2),

  best_book_over VARCHAR(50),
  best_book_odds_over INT,
  best_book_under VARCHAR(50),
  best_book_odds_under INT,

  my_best_book_over VARCHAR(50),
  my_best_book_odds_over INT,
  my_best_book_under VARCHAR(50),
  my_best_book_odds_under INT,

  implied_prob_over_with_vig DECIMAL(5, 4),
  implied_prob_under_with_vig DECIMAL(5, 4),
  implied_prob_over_devigged DECIMAL(5, 4),
  implied_prob_under_devigged DECIMAL(5, 4),

  bet_rule VARCHAR(255),
  bet_on TINYINT(1) DEFAULT 0,
  bet_on_side VARCHAR(20),

  wager DECIMAL(10, 2),

  winning_side VARCHAR(20),
  actual_total INT,
  bet_win_or_lose VARCHAR(20),
  profit DECIMAL(10, 2),

  game_date DATE NOT NULL,
  season INT NOT NULL,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  UNIQUE KEY unique_game (game_id),
  INDEX idx_game_date (game_date),
  INDEX idx_season (season),
  INDEX idx_bet_on (bet_on)
);
