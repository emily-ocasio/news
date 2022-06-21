DROP VIEW IF EXISTS view_shr;
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


