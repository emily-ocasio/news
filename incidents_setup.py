"""
 Set up DuckDB views for incident extraction from gptVictimJson (SQLite).
"""
from pymonad import Run, pure, put_line, ask, run_duckdb, sql_script, \
    sql_query, sql_exec, SQL
from menuprompts import NextStep

# --- DuckDB SQL ---
CREATE_STG_ARTICLE_INCIDENTS_SQL = SQL(r"""
CREATE OR REPLACE VIEW stg_article_incidents AS
WITH base AS (
  SELECT
    a.RecordId AS article_id,
    a.Publication AS city_id,
    a.PubDate AS publish_date,
    a.Title AS article_title,
    a.FullText AS article_text,

    CAST(i.key AS INTEGER) AS incident_idx,
    i.value AS incident_json,

    -- raw ints (may contain 0)
    try_cast(json_extract_string(i.value,'$.year')  AS INTEGER) AS year_raw,
    try_cast(json_extract_string(i.value,'$.month') AS INTEGER) AS month_raw,
    try_cast(json_extract_string(i.value,'$.day')   AS INTEGER) AS day_raw,

    json_extract_string(i.value,'$.location')       AS location_raw,
    json_extract_string(i.value,'$.circumstance')   AS circumstance,
    json_extract_string(i.value,'$.weapon')         AS weapon,
    try_cast(json_extract_string(i.value,'$.offender_count') AS INTEGER) AS offender_count,
    json_extract_string(i.value,'$.offender_name')  AS offender_name,
    try_cast(json_extract_string(i.value,'$.offender_age')  AS INTEGER) AS offender_age,
    json_extract_string(i.value,'$.offender_sex')   AS offender_sex,
    json_extract_string(i.value,'$.offender_race')  AS offender_race,
    json_extract_string(i.value,'$.offender_ethnicity') AS offender_ethnicity,
    try_cast(json_extract_string(i.value,'$.victim_count') AS INTEGER) AS victim_count,
    json_extract_string(i.value,'$.summary')        AS summary
  FROM sqldb.articles_wp_subset a
  CROSS JOIN json_each(
    CASE
      WHEN a.gptVictimJson IS NOT NULL AND a.gptVictimJson <> '' AND json_valid(a.gptVictimJson)
        THEN a.gptVictimJson
      ELSE '[]'
    END
  ) AS i
),
-- Sanitize Y/M/D, then compute date fields
dates AS (
  SELECT
    *,
    -- Treat 0 or out-of-range values as NULL
    CASE WHEN month_raw BETWEEN 1 AND 12 THEN month_raw END AS month_sane,
    CASE WHEN day_raw   BETWEEN 1 AND 31 THEN day_raw   END AS day_sane
  FROM base
),
norm_dates AS (
  SELECT
    *,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN make_date(year_raw, month_sane, day_sane)
      ELSE NULL
    END AS incident_date,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, 1))
      WHEN year_raw IS NOT NULL AND month_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, 1, 1))
      ELSE NULL
    END AS event_start_day,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year_raw, month_sane, 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY))
      WHEN year_raw IS NOT NULL AND month_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year_raw, 1, 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY))
      ELSE NULL
    END AS event_end_day
  FROM dates
),
-- Normalization features
norm AS (
  SELECT
    *,
    ((event_start_day IS NOT NULL) AND (event_end_day IS NOT NULL)) AS has_incident_date,
    (event_end_day - event_start_day) AS interval_span_days,
    lower(regexp_replace(coalesce(summary, ''), '[^a-z0-9 ]', '', 'g')) AS summary_norm,
    regexp_extract(coalesce(location_raw, ''), '([A-Za-z][A-Za-z0-9]+(?:\s(St|Ave|Blvd|Rd|Street|Avenue))?)') AS street_token
  FROM norm_dates
)
SELECT
  article_id, city_id,
  incident_idx,
  publish_date, article_title, article_text,
  year_raw AS year, month_sane AS month, day_sane AS day,
  incident_date, event_start_day, event_end_day,
  has_incident_date, interval_span_days,
  location_raw, street_token,
  circumstance, weapon,
  offender_count, offender_name, offender_age,
  offender_sex, offender_race, offender_ethnicity,
  victim_count, summary, summary_norm
FROM norm;
""")

CREATE_INCIDENTS_CACHED_SQL = SQL(r"""
CREATE OR REPLACE TABLE incidents_cached AS
SELECT * FROM stg_article_incidents;
""")

COUNT_VIEW_SQL    = SQL("SELECT COUNT(*) AS n FROM stg_article_incidents;")
COUNT_CACHE_SQL   = SQL("SELECT COUNT(*) AS n FROM incidents_cached;")
PREVIEW_CACHE_SQL = SQL(r"""
SELECT article_id, incident_idx, year, month, day, incident_date,
       event_start_day, event_end_day, weapon, circumstance,
       substr(summary, 1, 80) AS summary_snip
FROM incidents_cached
LIMIT 10;
""")


def build_incident_views() -> Run[NextStep]:
    """
    Steps:
      1) In SQLite (already under run_sqlite): rebuild filtered subset table.
      2) In DuckDB: attach SQLite, rebuild view from subset, materialize cache.
      3) Print counts and preview. Return to menu.
    Requires env:
      env["db_path"]      -> SQLite path
      env["duckdb_path"]  -> DuckDB persistent DB path
    """
    return ask() >> (lambda env:
        # --- 1) Rebuild subset in SQLite
        # (this runs in the existing run_sqlite context) ---
        put_line("[I] Rebuilding SQLite subset table articles_wp_subset…") ^
        sql_exec(SQL("DROP TABLE IF EXISTS articles_wp_subset;")) ^
        sql_exec(SQL(r"""
            CREATE TABLE articles_wp_subset AS
            SELECT RecordId, Publication, PubDate, Title, FullText, gptVictimJson, LastUpdated
            FROM articles
            WHERE Dataset='CLASS_WP' AND gptClass='M';
        """)) ^
        sql_exec(SQL(
            "CREATE INDEX IF NOT EXISTS idx_articles_wp_subset_pk "
            "ON articles_wp_subset(RecordId);")) ^
        sql_query(SQL("SELECT COUNT(*) AS n FROM articles_wp_subset;")) >> \
            (lambda rows:
            put_line(f"[I] SQLite subset rows: {rows[0]['n']}")
        ) ^

        # --- 2) Build DuckDB view + materialized cache from the subset ---
        put_line("[I] Building DuckDB view and materialized cache…") ^
        run_duckdb(
            env.get("duckdb_path", "news.duckdb"),
            (
                sql_script(CREATE_STG_ARTICLE_INCIDENTS_SQL) ^
                sql_query(COUNT_VIEW_SQL) >> (lambda rows:
                    put_line(f"[I] View rows: {rows[0]['n']}")
                ) ^
                sql_script(CREATE_INCIDENTS_CACHED_SQL) ^
                sql_query(COUNT_CACHE_SQL) >> (lambda rows:
                    put_line(f"[I] incidents_cached rows: {rows[0]['n']}")
                ) ^
                sql_query(PREVIEW_CACHE_SQL) >> (lambda rows:
                    put_line("[I] Preview incidents_cached (first 10):") ^
                    put_line("\n".join(
                        f"  id={r['article_id']} inc={r['incident_idx']} "
                        f"date={r['incident_date']} weapon={r['weapon']} "
                        f"circ={r['circumstance']} "
                        f"summary='{r['summary_snip']}'" for r in rows
                    ))
                )
            ),
            attach_sqlite_path=env["db_path"],
        ) ^
        put_line("[I] Done.") ^
        pure(NextStep.CONTINUE)
    )
