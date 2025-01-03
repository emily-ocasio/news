"""
Calculation functions that return sql statements
"""
def unverified_articles_sql() -> tuple[str, str]:
    """
    SQL statement to return articles not yet labeled for given date
    """
    sql = f"""
        FROM (
            {article_type_join_sql()}
            WHERE a.Pubdate = ?
            AND a.Status IS NULL
            GROUP BY a.RecordId
            HAVING GoodTypes > 0
        )
    """
    return "SELECT COUNT(*) " + sql, "SELECT * " + sql


def retrieve_types_sql() -> str:
    """
    SQL statement to return list of types for given article
    """
    return """
        SELECT e.TypeDesc desc 
        FROM articletypes t 
        JOIN articleenum e ON e.TypeId = t.TypeId 
        WHERE t.RecordId= ?
    """


def verify_article_sql_old() -> str:
    """
    SQL statement to update specific article with given label
    """
    return """
        UPDATE articles
        SET Status = ?
        WHERE RecordId = ?
    """


def verify_article_sql() -> str:
    """
    SQL statement to update specific article with given label
    """
    return """
        UPDATE articles
        SET Status = ?,
        LastUpdated = ?
        WHERE RecordId = ?
    """


def article_type_join_sql(index: str = "", extract: bool = False) -> str:
    """
    Initial portion of SQL statement joining articles with types
        and aggregating via binary encoding all the possible types
    """
    indexed_sql = "" if not index else f"INDEXED BY {index}"
    extract_sql = ("" if not extract
                    else ", t2.Extract, t2.SmallExtract, s.Victim, t2.Human")
    return f"""
        SELECT 
            a.*, 
            SUM(IIF(t.TypeID IN (7,8,9,10,12,13,19,21), 0, 1)) AS GoodTypes,
            SUM(1 << t.TypeId) as BinaryTypes
            {extract_sql}
        FROM articles a {indexed_sql}
        JOIN articletypes t
        ON a.RecordId = t.RecordId
    """


def verified_articles_sql() -> str:
    """
    SQL Statement to return articles with from a specified dataset and label
    """
    return article_type_join_sql() + """
        WHERE Status = ?
        AND Dataset = ?
        GROUP BY a.RecordId
    """


def all_articles_sql() -> str:
    """
    SQL statement to return all articles from a specific dataset
    """
    return article_type_join_sql() + """
        WHERE Dataset = ?
        GROUP BY a.RecordId
    """


def single_article_sql():
    """
    SQL statement to return a specific article by its recordId
    """
    return article_type_join_sql() + """
        WHERE a.RecordId = ?
        GROUP BY a.RecordId
    """


def single_article_with_extracts_sql():
    """
    SQL statement to return a specific article by its recordId
        joined with topics to provide specific extracts referring
        to a specific homicide
    """
    return article_type_join_sql(extract = True) + """
        INNER JOIN topics t2
        ON t2.RecordId = a.RecordId
        INNER JOIN shr s
        on t2.ShrId = s."Index"
        WHERE t2.ShrId = ?
        AND t2.RecordId = ?
        GROUP BY a.RecordId
        ORDER BY a.RecordId
    """


def passed_articles_sql() -> str:
    """
    SQL Statement to return articles with status 'P' (passed)
    """
    return article_type_join_sql() + """
        WHERE a.Status = 'P'
        GROUP BY a.RecordId
        ORDER BY a.PubDate
    """


def articles_to_classify_sql():
    """
    SQL statement to return articles to auto-classify based on date priority
    """
    return article_type_join_sql() + """
        WHERE PubDate IN (
            SELECT PubDate p2
            FROM dates
            WHERE Complete = 0
            AND p2 IN (
                SELECT DISTINCT PubDate p3
                FROM articles
                WHERE Dataset = "NOCLASS_WP"
                )
            ORDER BY Priority
            LIMIT ?
        )
        AND a.Dataset = "NOCLASS_WP"
        AND a.AutoClass IS NULL
        GROUP BY a.RecordId
    """


def articles_to_assign_sql():
    """
    SQL statement to return verified articles for assignment
    """
    return article_type_join_sql() + """
         WHERE a.Dataset = "CLASS"
         AND a.Status = "M"
         AND a.AssignStatus IS NULL
         AND a.PubDate IN (
             SELECT PubDate
             FROM dates
             WHERE PubDate IN (
                 SELECT DISTINCT PubDate
                 FROM articles
                 WHERE Dataset = "CLASS"
                 AND Status = "M"
                 AND AssignStatus IS NULL
             )
             ORDER BY Priority
             LIMIT ?
         )
         GROUP BY a.RecordId
         ORDER BY a.PubDate, a.RecordId
    """


def articles_to_assign_by_year_sql():
    """
    SQL Statement to return verified articles for assignment
        Instead of priority filter by year
    """
    return article_type_join_sql() + """
        WHERE a.Dataset = 'CLASS'
        AND a.Status = 'M'
        AND a.AssignStatus IS NULL
        AND a.PubDate >= ? 
        AND a.PubDate <= ?
        GROUP BY a.RecordId
        ORDER BY a.PubDate, a.RecordId
    """


def articles_humanizing_group_sql() -> str:
    """
    SQL Statement to select group of articles for humaninizing test
    Articles are based on homicide victim randomized groups
    Assumes homicides are each assigned to single articles
    """
    return article_type_join_sql() + """
        WHERE a.RecordId IN (
            SELECT RecordId
            FROM topics t
            INNER JOIN assigned a2
            ON t.ShrId = a2.ShrId
            WHERE a2.GroupSet = ?
        )
        GROUP BY a.RecordId
        ORDER BY a.RecordId
    """


def articles_to_reclassify_sql():
    """
    SQL statement to return auto-classified articles for reclassification
    Used when given a number of dates to grab auto-classified articles
        in order to verify whether they are in fact true positives
    """
    return """
        SELECT a.*
        FROM articles a
         WHERE a.Dataset = "CLASS"
         AND a.Status IS NULL
         AND a.Autoclass = "M"
         AND a.PubDate IN (
             SELECT PubDate
             FROM dates
             WHERE PubDate IN (
                 SELECT DISTINCT PubDate
                 FROM articles
                 WHERE Dataset = "CLASS"
                 AND Status IS NULL
                 AND Autoclass = "M"
             )
             ORDER BY Priority
             LIMIT ?
         )
         ORDER BY a.PubDate, a.RecordId
    """


def articles_to_reclassify_by_year_sql() -> str:
    """
    SQL statemnet to return auto-classified articles for reclassification
        given a particular year or set of years
    Used when selecting to review auto-classified articles and particularly
        in order to separate groups and have multiple users simultaneously
        reclassifying
    """
    return """
        SELECT a.*
        FROM articles a
        WHERE a.Dataset = 'CLASS'
        AND a.Status IS NULL
        AND a.AutoClass = 'M'
        AND a.PubDate >= ?
        AND a.PubDate <= ?
        ORDER BY a.PubDate, a.RecordId
    """


def classify_sql():
    """
    SQL statement to update auto-classification of a single article
    """
    return """
        UPDATE articles
        SET AutoClass = ?,
        Dataset = "CLASS_WP"
        WHERE RecordId = ?
    """


def assign_status_sql_old():
    """
    SQL statement to update assignment status for a single article
    """
    return """
        UPDATE articles
        SET AssignStatus = ?
        WHERE RecordId = ?
    """


def assign_status_sql():
    """
    SQL statement to update assignment status for a single article
    """
    return """
        UPDATE articles
        SET AssignStatus = ?,
        LastUpdated = ?
        WHERE RecordId = ?
    """


def update_note_sql_old() -> str:
    """
    SQL statement to update note in specific article
    """
    return """
        UPDATE articles
        SET Notes = ?
        WHERE RecordID = ?
    """


def update_note_sql() -> str:
    """
    SQL statement to update note in specific article
    """
    return """
        UPDATE articles
        SET Notes = ?,
        LastUpdated = ?
        WHERE RecordID = ?
    """


def cleanup_sql() -> str:
    """
    SQL Statement to update dates that have been completely autoclassified
    """
    return """
        WITH datelist AS MATERIALIZED (
            SELECT DISTINCT PubDate
            FROM articles
            INDEXED BY Dataset
            WHERE Dataset = "CLASS_WP"
            AND Pubdate NOT IN 
            (
                SELECT DISTINCT PubDate
                FROM articles
                INDEXED BY Dataset
                WHERE Dataset == "NOCLASS_WP"
            )
        )
        UPDATE dates
        SET Complete = 1
        WHERE PubDate IN (SELECT PubDate FROM datelist)
    """


def homicides_by_month_sql(clause: str) -> str:
    """
    SQL Statement to retrieve homicides based on year-month
    """
    return f"""
        SELECT ROW_NUMBER() OVER (ORDER BY Agency, Inc) AS n, *
        FROM view_shr
        WHERE YearMonth = ?
        AND {clause}
    """


def homicides_by_victim_sql() -> str:
    """
    SQL Statement to retrieve homicides based on victim name
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY YearMonth, Agency, Inc) AS n, *
        FROM view_shr
        WHERE Victim LIKE '%' || ? || '%'
    """


def homicides_by_county_sql() -> str:
    """
    SQL Statement to retrieve all homicides from a county
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY YearMonth, Agency, Inc) AS n, *
        FROM view_shr
        WHERE County LIKE ?
    """


def homicides_assigned_by_article_sql() -> str:
    """
    SQL Statement to retrieve homicides already assigned
        to a specific article
    Also retrieves result of manual and automatic humanizing
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY Agency, Inc) AS k, v.*,
            IFNULL(t.HumanManual, '') AS HM, IFNULL(t.Human, '') AS H,
            IFNULL(t.Extract, '') AS Extract,
            IFNULL(t.SmallExtract, '') AS SmallExtract
        FROM view_shr v
        INNER JOIN topics t
        ON t.ShrId = v.Id
        WHERE RecordId = ?
    """


def homicides_by_group_sql() -> str:
    """
    SQL Statement to retrieve homicides based on priority group
    Also computes humanizing status
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY Victim) AS k, v.*,
            MAX(IIF(t.Human = 3,3,IIF(t.Human>0,1,0))) AS H,
            MAX(IIF(t.HumanManual = 3,3,IIF(t.HumanManual>0,1,0))) AS HM
        FROM view_shr v
        INNER JOIN topics t
        ON t.ShrId = v.Id
        WHERE v.Id IN
            (
                SELECT ShrId
                FROM assigned a
                WHERE a.GroupSet = ?
            )
        GROUP BY v.Id
    """


def homicide_refreshed_sql() -> str:
    """
    SQL Statement to retrieve single homicide again to refresh
        humanizing status
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY Victim) AS k, v.*,
            MAX(IIF(t.Human = 3,3,IIF(t.Human>0,1,0))) AS H,
            MAX(IIF(t.HumanManual = 3,3,IIF(t.HumanManual>0,1,0))) AS HM
        FROM view_shr v
        INNER JOIN topics t
        ON t.ShrId = v.Id
        WHERE v.Id = ?
        GROUP BY v.Id
    """



def articles_from_homicide_sql() -> str:
    """
    SQL Statement to retrieve articles that have been already assigned to
        a particular homicide
    """
    return article_type_join_sql(extract = True) + """
        INNER JOIN topics t2
        ON t2.RecordId = a.RecordId
        INNER JOIN shr s
        on t2.ShrId = s."Index"
        WHERE t2.ShrId = ?
        GROUP BY a.RecordId
        ORDER BY a.RecordId
    """


def assign_homicide_victim_sql_old() -> str:
    """
    SQL Statement (transaction) to add assignment of homicide
        to a specific article and also adjust the victim name
    """
    return """
            INSERT OR IGNORE INTO topics
            (ShrId, RecordId)
            VALUES (?, ?);
            UPDATE shr
            SET Victim = ?
            WHERE "index" = ?
    """


def assign_homicide_victim_sql() -> str:
    """
    SQL Statement (transaction) to add assignment of homicide
        to a specific article and also adjust the victim name
    """
    return """
            INSERT OR IGNORE INTO topics
            (ShrId, RecordId, LastUpdated)
            VALUES (?, ?, ?);
            UPDATE shr
            SET Victim = ?
            WHERE "index" = ?
    """


def assign_homicide_sql_old(repeat:int = 1) -> str:
    """
    SQL Statement to add assignment of homicide
        without changing victim's name
    Allows for multi-row insert (repeat number of rows inserted)
    """
    return """
        INSERT OR IGNORE INTO topics
        (ShrId, RecordId)
        VALUES """ + ' , '.join(('(?, ?)',) * repeat)


def assign_homicide_sql(repeat:int = 1) -> str:
    """
    SQL Statement to add assignment of homicide
        without changing victim's name
    Allows for multi-row insert (repeat number of rows inserted)
    """
    return """
        INSERT OR IGNORE INTO topics
        (ShrId, RecordId, LastUpdated)
        VALUES """ + ' , '.join(('(?, ?, ?)',) * repeat)


def unassign_homicide_sql() -> str:
    """
    SQL Statement to un-assign (delete from assignment list)
        a particular homicide previously assigned to an article
    """
    return """
        DELETE FROM topics
        WHERE ShrId = ?
        AND RecordId = ?
    """


def manual_humanizing_sql() -> str:
    """
    SQL Statement to set the manual (human ground truth)
        humanizing value for a particular victim in an article
    """
    return """
        UPDATE topics
        SET HumanManual = ?
        WHERE ShrId = ?
        AND RecordId = ?
    """


def gpt3_humanizing_sql() -> str:
    """
    SQL Statement to set the gpt3
        humanizing value for a particular victim in an article
    """
    return """
        UPDATE topics
        SET Human = ?
        WHERE ShrId = ?
        AND RecordId = ?;
        INSERT INTO gptAttempts
        (RecordId, ShrId, Human, HumanManual, PreArticle, PostArticle,
            Prompt, Response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """


def gpt3_extract_sql() -> str:
    """
    SQL Statement to save the GPT-3 extracted text
        specific to a particular victim
    """
    return """
        UPDATE topics
        SET Extract = ?
        WHERE ShrId = ?
        AND RecordId = ?;
        INSERT INTO gptAttempts
        (RecordId, ShrId, Human, HumanManual, PreArticle, PostArticle,
            Prompt, Response)
        VALUES (?,?,?,?,?,?,?,?)
    """


def gpt3_small_extract_sql() -> str:
    """
    SQL Statement to save the GPT-3 extracted text
        specific to a particular victim
    """
    return """
        UPDATE topics
        SET SmallExtract = ?
        WHERE ShrId = ?
        AND RecordId = ?;
        INSERT INTO gptAttempts
        (RecordId, ShrId, Human, HumanManual, PreArticle, PostArticle,
            Prompt, Response)
        VALUES (?,?,?,?,?,?,?,?)
    """

def gpt_homicide_class_sql() -> str:
    """
    SQL Statement to set the gpt
        homicide class for an article
    """
    return """
        UPDATE articles
        SET gptClass = ?
        WHERE RecordId = ?
    """

def gpt_victims_sql() -> str:
    """
    SQL Statement to set the gpt json text for the victims extract
    """
    return """
        UPDATE articles
        SET gptVictimJson = ?
        WHERE RecordId = ?
    """

def articles_to_filter_sql() -> str:
    """
    SQL statement to return articles to filter based on a limit
    Only considers articles within the dataset called 'CLASSTRAIN'
    """
    return """
        SELECT * FROM articles
        WHERE Dataset = 'CLASS_WP'
        AND AutoClass = 'M'
        AND gptClass IS NULL
        ORDER BY PubDate
        LIMIT ?
    """

def articles_by_victim_sql() -> str:
    """
    SQL statement to return articles by victim id
    """
    return article_type_join_sql() + """
        INNER JOIN topics top ON a.RecordId = top.RecordId
        WHERE top.ShrId = ?
        GROUP BY a.RecordId
        ORDER BY a.PubDate, a.RecordId
    """
