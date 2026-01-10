CREATE TABLE articles (
    RecordId INTEGER PRIMARY KEY,
    Timestamp INTEGER NOT NULL,
    Title TEXT NOT NULL,
    Publication INTEGER NOT NULL,
    PubDate TEXT NOT NULL,
    Page TEXT,
    Pagination TEXT,
    FullText TEXT,
    Abstract TEXT 
, Status TEXT, Dataset TEXT, AutoClass TEXT, ManualClass TEXT, Notes TEXT, AssignStatus TEXT, LastUpdated TEXT, gptClass TEXT, gptVictimJson TEXT, Guid TEXT);
CREATE TABLE authors (
    OrigForm TEXT NOT NULL,
    RecordId INTEGER NOT NULL,
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId)
);
CREATE TABLE articleenum (
    TypeId INTEGER PRIMARY KEY,
    TypeDesc TEXT NOT NULL
);
CREATE TABLE articletypes (
    TypeId INTEGER NOT NULL,
    RecordId INTEGER NOT NULL,
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId),
    FOREIGN KEY(TypeId) REFERENCES articleenum(TypeId)
);
CREATE TABLE sqlite_stat1(tbl,idx,stat);
CREATE TABLE dates (
    PubDate TEXT NOT NULL,
    Priority INTEGER NOT NULL DEFAULT 0,
    Complete INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX PubDate ON articles(PubDate);
CREATE INDEX ArticleType on articletypes(RecordId);
CREATE INDEX Status ON articles(Status);
CREATE INDEX Dataset ON articles(Dataset, PubDate);
CREATE INDEX DataStatus ON ARTICLES (Dataset, Status, PubDate);
CREATE TABLE IF NOT EXISTS "shr" (
"index" INTEGER,
  "ID" TEXT,
  "CNTYFIPS" TEXT,
  "Ori" TEXT,
  "State" TEXT,
  "Agency" TEXT,
  "Agentype" TEXT,
  "Source" TEXT,
  "Solved" TEXT,
  "Year" INTEGER,
  "StateName" TEXT,
  "Month" TEXT,
  "Incident" INTEGER,
  "ActionType" TEXT,
  "Homicide" TEXT,
  "Situation" TEXT,
  "VicAge" INTEGER,
  "VicSex" TEXT,
  "VicRace" TEXT,
  "VicEthnic" TEXT,
  "OffAge" INTEGER,
  "OffSex" TEXT,
  "OffRace" TEXT,
  "OffEthnic" TEXT,
  "Weapon" TEXT,
  "Relationship" TEXT,
  "Circumstance" TEXT,
  "Subcircum" TEXT,
  "VicCount" INTEGER,
  "OffCount" INTEGER,
  "FileDate" REAL,
  "MSA" TEXT,
  "YearMonth" TEXT
, Victim TEXT);
CREATE INDEX "ix_shr_index"ON "shr" ("index");
CREATE INDEX "shr_id" on shr (id);
CREATE INDEX shr_date ON shr(YearMonth);
CREATE INDEX DataStatus2 ON articles (Dataset, Status, Autoclass, PubDate);
CREATE TABLE topics (
    ShrId INTEGER NOT NULL,
    RecordId INTEGER NOT NULL, LastUpdated TEXT, Human TEXT, HumanManual TEXT, Extract TEXT, SmallExtract TEXT,
    UNIQUE(ShrId, RecordId)
    FOREIGN KEY(RecordId) REFERENCES articles(RecordId),
    FOREIGN KEY(ShrId) REFERENCES shr("index")
);
CREATE INDEX topics_shrid ON topics (ShrId);
CREATE INDEX topics_recordid ON topics (RecordId);
CREATE INDEX shr_victim on shr(Victim);
CREATE INDEX AssignStatus ON articles (AssignStatus);
CREATE INDEX AssignStatus2 ON articles (Dataset, Status, AssignStatus, PubDate);
CREATE INDEX articles_lastupdated ON articles (LastUpdated);
CREATE INDEX topics_lastupdated ON topics (LastUpdated);
CREATE TABLE gptAttempts (
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
CREATE INDEX attempts 
    ON gptAttempts(ShrId, RecordId, PostArticle, PreArticle);
CREATE VIEW view_shr AS
    SELECT
        "index" AS Id,
        RTRIM(CNTYFIPS, ', MA') AS County,
        REPLACE(
            REPLACE(
                REPLACE(Agency, 'State Police: ', ''), 
            'Bay Transist Authority: ', ''),
        ' Met District Commission', '')
        AS Agency,
        Incident AS Inc,
        VicCount+1 AS VCnt,
        VicAge AS VAge,
        SUBSTR(VicSex,1,4) AS VSex,
        SUBSTR(VicRace,1,4) AS VRace,
        SUBSTR(VicEthnic,1,4) AS VHisp,
        IIF(INSTR(Situation, 'unknown'),0,OffCount+1) AS OCnt,
        OffAge AS OAge,
        SUBSTR(OffSex,1,4) AS OSex,
        SUBSTR(OffRace,1,4) AS ORace,
        SUBSTR(OffEthnic,1,4) AS OHisp,
        SUBSTR(Weapon,1,7) AS Weapon,
        SUBSTR(REPLACE(Relationship, 'Relationship ', ''),1,11) AS Relationship,
        SUBSTR(Circumstance,1,20) as Circumstance,
        SUBSTR(MSA,1,5) as MSA,
        Victim AS Victim,
        YearMonth AS YearMonth
    FROM shr
/* view_shr(Id,County,Agency,Inc,VCnt,VAge,VSex,VRace,VHisp,OCnt,OAge,OSex,ORace,OHisp,Weapon,Relationship,Circumstance,MSA,Victim,YearMonth) */;
CREATE TABLE assigned (
    ShrId INTEGER NOT NULL,
    ArticleCount INTEGER NOT NULL,
    GroupSet TEXT,
    Priority INTEGER,
    GroupPriority INTEGER,
    FOREIGN KEY(ShrId) REFERENCES shr("index")
);
CREATE INDEX assigned_shrid 
    ON assigned(ShrId, GroupSet, GroupPriority);
CREATE VIEW assignments AS
    SELECT
        s.*,
        COUNT(t.RecordId) AssignCount,
        MAX(IIF(t.Human = 3,1,IIF(t.Human>0,0,NULL))) AS Humanized
    FROM shr s
    LEFT JOIN topics t ON s."index" = t.ShrId
    GROUP BY s."index"
/* assignments("index",ID,CNTYFIPS,Ori,State,Agency,Agentype,Source,Solved,Year,StateName,Month,Incident,ActionType,Homicide,Situation,VicAge,VicSex,VicRace,VicEthnic,OffAge,OffSex,OffRace,OffEthnic,Weapon,Relationship,Circumstance,Subcircum,VicCount,OffCount,FileDate,MSA,YearMonth,Victim,AssignCount,Humanized) */;
CREATE TABLE gptVictims (
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
CREATE TRIGGER increment_victim_num
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
CREATE TABLE incidents (
    IncidentNum INTEGER PRIMARY KEY AUTOINCREMENT,
    RecordId INTEGER NOT NULL,
    Year INTEGER,
    Month INTEGER,
    Day INTEGER,
    Location TEXT,
    Circumstance TEXT,
    Weapon TEXT,
    OffCount INTEGER,
    OffName TEXT,
    OffAge INTEGER,
    OffSex TEXT,
    OffRace TEXT,
    OffEthnic TEXT,
    MatchYear INTEGER,
    MatchMonth INTEGER,
    MatchIncidentNum INTEGER,
    Summary TEXT,
    FOREIGN KEY (RecordId) REFERENCES articles(RecordId)
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE victims (
    VictimId INTEGER PRIMARY KEY AUTOINCREMENT,
    IncidentNum INTEGER NOT NULL,
    VicName TEXT,
    VicAge INTEGER,
    VicSex TEXT,
    VicRace TEXT,
    VicEthnic TEXT,
    Relationship TEXT,
    FOREIGN KEY (IncidentNum) REFERENCES incidents(IncidentNum)
);
CREATE INDEX incidents_recordid ON incidents(RecordId);
CREATE INDEX victims_incidentnum ON victims(IncidentNum);
CREATE TABLE gptResults (
    ResultId INTEGER PRIMARY KEY,
    RecordId INTEGER NOT NULL,
    UserName TEXT NOT NULL,
    TimeStamp TEXT NOT NULL,
    PromptKey TEXT,
    PromptId TEXT,
    PromptVersion TEXT,
    Variables TEXT,
    Model TEXT,
    FormatType TEXT,
    OutputJson TEXT,
    Reasoning TEXT,
    TotalInputTokens INTEGER,
    CachedInputTokens INTEGER,
    TotalOutputTokens INTEGER,
    ReasoningTokens INTEGER,
    CostPerThousand REAL,
    FOREIGN KEY (RecordId) REFERENCES articles(RecordId)
);
CREATE INDEX gptResults_recordid ON gptResults(RecordId);
CREATE TABLE articles_wp_subset(
  RecordId INT,
  Publication INT,
  PubDate TEXT,
  Title TEXT,
  FullText TEXT,
  gptVictimJson TEXT,
  LastUpdated TEXT
);
CREATE INDEX idx_articles_wp_subset_pk ON articles_wp_subset(RecordId);