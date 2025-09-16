"""
 Set up DuckDB views for incident extraction from gptVictimJson (SQLite).
"""
from pymonad import Run, pure, put_line, ask, run_duckdb, sql_script, \
    sql_query, SQL
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

    -- index of element in the root array (0-based)
    CAST(i.key AS INTEGER) AS incident_idx,
    i.value AS incident_json,

    try_cast(json_extract_string(i.value, '$.year')  AS INTEGER) AS year,
    try_cast(json_extract_string(i.value, '$.month') AS INTEGER) AS month,
    try_cast(json_extract_string(i.value, '$.day')   AS INTEGER) AS day,
    json_extract_string(i.value, '$.location')       AS location_raw,
    json_extract_string(i.value, '$.circumstance')   AS circumstance,
    json_extract_string(i.value, '$.weapon')         AS weapon,
    try_cast(json_extract_string(i.value, '$.offender_count') AS INTEGER) AS offender_count,
    json_extract_string(i.value, '$.offender_name')  AS offender_name,
    try_cast(json_extract_string(i.value, '$.offender_age')  AS INTEGER) AS offender_age,
    json_extract_string(i.value, '$.offender_sex')   AS offender_sex,
    json_extract_string(i.value, '$.offender_race')  AS offender_race,
    json_extract_string(i.value, '$.offender_ethnicity') AS offender_ethnicity,
    try_cast(json_extract_string(i.value, '$.victim_count') AS INTEGER) AS victim_count,
    json_extract_string(i.value, '$.summary')        AS summary
  FROM (
    SELECT *
    FROM sqldb.articles
    WHERE Dataset = 'CLASS_WP' AND gptClass = 'M'
  ) a
  -- gptVictimJson is a ROOT ARRAY, so traverse it directly (no '$.incidents')
  CROSS JOIN json_each(
    CASE
      WHEN a.gptVictimJson IS NOT NULL
           AND a.gptVictimJson <> ''
           AND json_valid(a.gptVictimJson)
        THEN a.gptVictimJson
      ELSE '[]'  -- safe fallback: empty array yields 0 rows
    END
  ) AS i
),
-- Date derivations
dates AS (
  SELECT
    *,
    CASE
      WHEN year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
        THEN make_date(year, month, day)
      ELSE NULL
    END AS incident_date,
    CASE
      WHEN year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year, month, day))
      WHEN year IS NOT NULL AND month IS NOT NULL AND day IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year, month, 1))
      WHEN year IS NOT NULL AND month IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year, 1, 1))
      ELSE NULL
    END AS event_start_day,
    CASE
      WHEN year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year, month, day))
      WHEN year IS NOT NULL AND month IS NOT NULL AND day IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year, month, 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY))
      WHEN year IS NOT NULL AND month IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year, 1, 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY))
      ELSE NULL
    END AS event_end_day
  FROM base
),
-- Normalization features
norm AS (
  SELECT
    *,
    ((event_start_day IS NOT NULL) AND (event_end_day IS NOT NULL)) AS has_incident_date,
    (event_end_day - event_start_day) AS interval_span_days,
    lower(regexp_replace(coalesce(summary, ''), '[^a-z0-9 ]', '', 'g')) AS summary_norm,
    regexp_extract(coalesce(location_raw, ''), '([A-Za-z][A-Za-z0-9]+(?:\s(St|Ave|Blvd|Rd|Street|Avenue))?)') AS street_token
  FROM dates
)
SELECT
  article_id, city_id,
  incident_idx,
  publish_date, article_title, article_text,
  year, month, day,
  incident_date, event_start_day, event_end_day,
  has_incident_date, interval_span_days,
  location_raw, street_token,
  circumstance, weapon,
  offender_count, offender_name, offender_age,
  offender_sex, offender_race, offender_ethnicity,
  victim_count, summary, summary_norm
FROM norm;
""")

COUNT_SQL = SQL("SELECT COUNT(*) AS n FROM stg_article_incidents;")
PREVIEW_SQL = SQL(r"""
SELECT article_id, incident_idx, year, month, day, incident_date,
       event_start_day, event_end_day, weapon, circumstance,
       substr(summary, 1, 80) AS summary_snip
FROM stg_article_incidents
LIMIT 10;
""")


def build_incident_views() -> Run[NextStep]:
    """
    Controller: Build/refresh the DuckDB staging view `stg_article_incidents`.
    Reads from Reader env via `ask()`:
      env["db_path"]      -> path to SQLite DB (attached as `sqldb`)
      env["duckdb_path"]  -> persistent DuckDB catalog path
    """
    return ask() >> (lambda env:
        (
            put_line(
                "[incidents] building DuckDB view stg_article_incidents …") ^
            run_duckdb(
                env.get("duckdb_path", "news.duckdb"),
                # Program: create view, then count and preview
                (
                    sql_script(CREATE_STG_ARTICLE_INCIDENTS_SQL) ^
                    sql_query(COUNT_SQL) >> (lambda rows:
                        put_line(f"[incidents] rows: {rows[0]['n']}")
                    ) ^
                    sql_query(PREVIEW_SQL) >> (lambda rows:
                        put_line("[incidents] preview (first 10):") ^
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
            put_line("[incidents] done.") ^
            pure(NextStep.CONTINUE)
        )
    )
