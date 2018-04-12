create table if not exists  records (
  id integer primary key autoincrement,
  caption string not null,
  raw_path string not null,
  att_path string not null,
  rank TINYINT,
  created_at TIMESTAMP default (datetime('now', 'localtime')) ,
  updated_at TIMESTAMP,
  deleted_at TIMESTAMP
);
CREATE INDEX if not exists rank_index ON records (rank);