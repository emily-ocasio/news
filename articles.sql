CREATE TABLE IF NOT EXISTS articles (
    RecordId INTEGER PRIMARY KEY,
    Timestamp INTEGER NOT NULL,
    Title TEXT NOT NULL,
    Publication INTEGER NOT NULL,
    PubDate TEXT NOT NULL,
    Page TEXT,
    Pagination TEXT,
    FullText TEXT,
    Abstract TEXT,
    Status TEXT,
    Dataset TEXT,
    AutoClass TEXT,
    ManualClass TEXT,
    Notes TEXT,
    AssignStatus TEXT,
    LastUpdated TEXT,
    gptClass TEXT,
    gptVictimJson TEXT
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
    Extract TEXT,
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

CREATE TABLE IF NOT EXISTS assigned (
    ShrId INTEGER NOT NULL,
    ArticleCount INTEGER NOT NULL,
    GroupSet TEXT,
    Priority INTEGER,
    GroupPriority INTEGER,
    FOREIGN KEY(ShrId) REFERENCES shr("index")
);

CREATE TABLE IF NOT EXISTS gptVictims (
    RecordId INTEGER NOT NULL,
    VictimNum INTEGER NOT NULL,
    VicName TEXT,
    VicAge INTEGER,
    VicSex TEXT,
    VicRace TEXT,
    VicEthnic TEXT,
    OffName TEXT,
    OffAge INTEGER,
    OffSex TEXT,
    OffRace TEXT,
    OffEthnic TEXT,
    OffCount INTEGER,
    Relationship TEXT,
    Circumstance TEXT,
    Date TEXT,
    Extract TEXT,
    County TEXT,
    City TEXT,
    PRIMARY KEY (RecordId, VictimNum),
    FOREIGN KEY (RecordId) REFERENCES articles(RecordId)
);

CREATE TRIGGER IF NOT EXISTS increment_victim_num
AFTER INSERT ON gptVictims
FOR EACH ROW
BEGIN
    UPDATE gptVictims
    SET VictimNum = (
        SELECT COALESCE(MAX(StoryNum), 0) + 1
        FROM gptVictims
        WHERE RecordId = NEW.RecordId
    )
    WHERE rowid = NEW.rowid;
END;

CREATE INDEX IF NOT EXISTS attempts 
    ON gptAttempts(ShrId, RecordId, PostArticle, PreArticle);

CREATE INDEX IF NOT EXISTS topics_shrid ON topics(ShrId);
CREATE INDEX IF NOT EXISTS topics_recordid ON topics(RecordId);

CREATE INDEX IF NOT EXISTS "shr_id" ON shr(id);
CREATE INDEX IF NOT EXISTS shr_date ON shr(YearMonth);
CREATE INDEX IF NOT EXISTS shr_victim on shr(Victim);

CREATE INDEX IF NOT EXISTS assigned_shrid 
    ON assigned(ShrId, GroupSet, GroupPriority);
