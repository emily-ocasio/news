CREATE TABLE IF NOT EXISTS articles (
    RecordId INTEGER PRIMARY KEY,
    Timestamp INTEGER NOT NULL,
    Title TEXT NOT NULL,
    Publication INTEGER NOT NULL,
    PubDate TEXT NOT NULL,
    Page TEXT,
    Pagination TEXT,
    FullText TEXT,
    Abstract TEXT 
);

CREATE TABLE IF NOT EXISTS authors (
    OrigForm TEXT NOT NULL,
    RecordId INTEGER NOT NULL,
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId)
);

CREATE TABLE IF NOT EXISTS articleenum (
    TypeId INTEGER PRIMARY KEY,
    TypeDesc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articletypes (
    TypeId INTEGER NOT NULL,
    RecordId INTEGER NOT NULL,
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId),
    FOREIGN KEY(TypeId) REFERENCES articleenum(TypeId)
);

CREATE TABLE IF NOT EXISTS dates (
    PubDate TEXT NOT NULL,
    Priority INTEGER NOT NULL DEFAULT 0,
    Complete INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topics (
    ShrId INTEGER NOT NULL,
    RecordId INTEGER NOT NULL,
    LastUpdated TEXT,
    Human TEXT,
    HumanManual TEXT,
    UNIQUE(ShrId, RecordId),
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId),
    FOREIGN KEY(ShrId) REFERENCES shr("index")
);

CREATE TABLE IF NOT EXISTS gptAttempts (
    RecordId INTEGER NOT NULL,
    ShrId INTEGER NOT NULL,
    Human TEXT NOT NULL,
    HumanManual TEXT NOT NULL,
    PreArticle TEXT NOT NULL,
    PostArticle TEXT NOT NULL,
    Prompt TEXT NOT NULL,
    Response TEXT NOT NULL,
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId),
    FOREIGN KEY(ShrId) REFERENCES shr("index")
);

CREATE INDEX IF NOT EXISTS attempts 
    ON gptAttempts(ShrId, RecordId, PostArticle, PreArticle);

CREATE INDEX IF NOT EXISTS topics_shrid ON topics(ShrId);
CREATE INDEX IF NOT EXISTS topics_recordid ON topics(RecordId);

CREATE INDEX IF NOT EXISTS "shr_id" ON shr(id);
CREATE INDEX IF NOT EXISTS shr_date ON shr(YearMonth);
CREATE INDEX IF NOT EXISTS shr_victim on shr(Victim);