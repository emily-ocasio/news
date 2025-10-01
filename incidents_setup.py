"""
 Set up DuckDB views for incident extraction from gptVictimJson (SQLite).
"""
from pymonad import Run, pure, put_line, ask, run_duckdb, sql_script, \
    sql_query, sql_exec, SQL
from menuprompts import NextStep

# --- DuckDB SQL ---
# --- Victim-level extraction & feature engineering ---

# 1) Explode victim array -> one row per victim mention
CREATE_VICTIMS_CACHED_SQL = SQL(r"""
CREATE OR REPLACE TABLE victims_cached AS
SELECT
  a.article_id,
  a.city_id,
  a.publish_date,
  a.year, a.month, a.day,
  a.incident_idx,
  CAST(v.key AS INTEGER) AS victim_idx,
  json_extract_string(v.value, '$.victim_name')      AS victim_name_raw,
  try_cast(json_extract_string(v.value, '$.victim_age') AS INTEGER) AS victim_age_raw,
  json_extract_string(v.value, '$.victim_sex')       AS victim_sex,
  json_extract_string(v.value, '$.victim_race')      AS victim_race,
  json_extract_string(v.value, '$.victim_ethnicity') AS victim_ethnicity,

  a.incident_date, a.event_start_day, a.event_end_day,
  a.weapon, a.circumstance,

  -- keep raw offender_name so we can normalize it with the same logic as victims
  a.offender_name,

  try_cast(json_extract_string(v.value, '$.victim_count') AS INTEGER) AS victim_count,

  concat_ws(':', CAST(a.article_id AS VARCHAR), CAST(a.incident_idx AS VARCHAR), CAST(CAST(v.key AS INTEGER) AS VARCHAR)) AS victim_row_id
FROM incidents_cached a
CROSS JOIN json_each(CAST(a.incident_json AS JSON), '$.victim') AS v
WHERE json_type(CAST(a.incident_json AS JSON), '$.victim') = 'ARRAY';
""")

# 2) Normalize names + sanitize age,
# add coarse age bucket + parsed forename/surname
CREATE_VICTIMS_CACHED_ENH_SQL = SQL(r"""
CREATE OR REPLACE TABLE victims_cached_enh AS
WITH norm AS (
  SELECT
    v.*,
    trim(
      lower(
        regexp_replace(
          coalesce(victim_name_raw, ''),
          '[^A-Za-z''\- ]',
          ' ',
          'g'
        )
      )
    ) AS name_clean,
    trim(
      lower(
        regexp_replace(
          coalesce(offender_name, ''),
          '[^A-Za-z''\- ]',
          ' ',
          'g'
        )
      )
    ) AS offender_name_clean
  FROM victims_cached v
),
strip AS (
  SELECT
    *,
    regexp_replace(name_clean, '^(?i)(mr|mrs|ms|miss|dr|rev|ofc|officer)\s+', '', 'g') AS name_no_title,
    regexp_replace(offender_name_clean, '^(?i)(mr|mrs|ms|miss|dr|rev|ofc|officer)\s+', '', 'g') AS offender_no_title
  FROM norm
),
strip2 AS (
  SELECT
    *,
    regexp_replace(name_no_title, '\s+(?i)(jr|sr|ii|iii|iv)\.?$', '', 'g') AS name_core,
    regexp_replace(offender_no_title, '\s+(?i)(jr|sr|ii|iii|iv)\.?$', '', 'g') AS offender_name_core
  FROM strip
),
parts AS (
  SELECT
    *,
    CASE
      WHEN victim_name_raw IS NULL THEN NULL
      WHEN name_core LIKE '% %' THEN regexp_extract(name_core, '^([a-z''\-]+)', 1)
      ELSE NULL
    END AS victim_forename_norm,
    CASE
      WHEN victim_name_raw IS NULL THEN NULL
      WHEN name_core LIKE '% %' THEN
        CASE
          WHEN regexp_matches(name_core, '(?i)(?:^|\s)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+$')
            THEN regexp_extract(name_core, '((?i)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+)$', 1)
          ELSE regexp_extract(name_core, '([a-z''\-]+)$', 1)
        END
      ELSE NULL
    END AS victim_surname_norm,
    CASE
      WHEN offender_name IS NULL THEN NULL
      WHEN offender_name_core LIKE '% %' THEN regexp_extract(offender_name_core, '^([a-z''\-]+)', 1)
      ELSE NULL
    END AS offender_forename_norm,
    CASE
      WHEN offender_name IS NULL THEN NULL
      WHEN offender_name_core LIKE '% %' THEN
        CASE
          WHEN regexp_matches(offender_name_core, '(?i)(?:^|\s)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+$')
            THEN regexp_extract(offender_name_core, '((?i)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+)$', 1)
          ELSE regexp_extract(offender_name_core, '([a-z''\-]+)$', 1)
        END
      ELSE NULL
    END AS offender_surname_norm
  FROM strip2
)
SELECT
  p.*,
  CASE
    WHEN victim_forename_norm IS NOT NULL AND victim_surname_norm IS NOT NULL
      THEN victim_forename_norm || ' ' || victim_surname_norm
  END AS victim_fullname_concat,
  CASE
    WHEN offender_forename_norm IS NOT NULL AND offender_surname_norm IS NOT NULL
      THEN offender_forename_norm || ' ' || offender_surname_norm
  END AS offender_fullname_concat,
  regexp_replace(name_core, '[^a-z ]', '', 'g') AS victim_name_norm,
  regexp_replace(offender_name_core, '[^a-z ]', '', 'g') AS offender_name_norm,
  CASE WHEN victim_age_raw BETWEEN 0 AND 120 THEN victim_age_raw END AS victim_age,
  CASE WHEN victim_age_raw BETWEEN 0 AND 120 THEN floor(victim_age_raw/5) END AS victim_age_bucket5,
  i.date_precision,
  i.midpoint_day
FROM parts p
LEFT JOIN incidents_cached i
  ON p.article_id = i.article_id
 AND p.incident_idx = i.incident_idx;
""")

COUNT_VICTIMS_SQL  = SQL("SELECT COUNT(*) AS n FROM victims_cached_enh;")
PREVIEW_VICTIMS_SQL = SQL(r"""
SELECT article_id, incident_idx, victim_idx, victim_row_id,
       victim_name_raw, victim_age, victim_sex, victim_race, victim_ethnicity
FROM victims_cached_enh
LIMIT 10;
""")
# --- Incident-level extraction & feature engineering ---
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
dates AS (
  SELECT
    *,
    CASE WHEN month_raw BETWEEN 1 AND 12 THEN month_raw END AS month_sane,
    CASE WHEN day_raw   BETWEEN 1 AND 31 THEN day_raw   END AS day_sane
  FROM base
),
norm_dates AS (
  SELECT
    *,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL THEN 'day'
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL THEN 'month'
      WHEN year_raw IS NOT NULL THEN 'year'
    END AS date_precision,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN make_date(year_raw, month_sane, day_sane)
    END AS incident_date,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, 1))
      WHEN year_raw IS NOT NULL AND month_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, 1, 1))
    END AS event_start_day,
    CASE
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
        THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
      WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year_raw, month_sane, 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY))
      WHEN year_raw IS NOT NULL AND month_sane IS NULL
        THEN date_diff('day', DATE '1970-01-01', (make_date(year_raw, 1, 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY))
    END AS event_end_day,
    -- Add midpoint_day for partial dates
    CASE
      WHEN event_start_day IS NOT NULL AND event_end_day IS NOT NULL
        THEN floor((event_start_day + event_end_day) / 2)
    END AS midpoint_day
  FROM dates
),
norm AS (
  SELECT
    *,
    ((event_start_day IS NOT NULL) AND (event_end_day IS NOT NULL)) AS has_incident_date,
    (event_end_day - event_start_day) AS interval_span_days,
    lower(regexp_replace(coalesce(summary, ''), '[^a-z0-9 ]', '', 'g')) AS summary_norm
  FROM norm_dates
)
SELECT
  article_id, city_id,
  incident_idx,
  CAST(incident_json AS JSON) AS incident_json,
  publish_date, article_title, article_text,
  year_raw AS year, month_sane AS month, day_sane AS day,
  incident_date, event_start_day, event_end_day, midpoint_day,
  date_precision,  -- Include precision in final output
  has_incident_date, interval_span_days,
  location_raw,
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
                # (optional) sanity check that column exists
                sql_query(SQL("PRAGMA table_info('incidents_cached')")) >> \
                  (lambda rows:
                    put_line("[I] incidents_cached columns: " +
                            ", ".join(str(r['name']) for r in rows))
                ) ^
                sql_script(CREATE_VICTIMS_CACHED_SQL) ^
                sql_script(CREATE_VICTIMS_CACHED_ENH_SQL) ^
                sql_query(COUNT_VICTIMS_SQL) >> (lambda rows:
                    put_line(f"[I] victims_cached_enh rows: {rows[0]['n']}")
                ) ^
                sql_query(PREVIEW_VICTIMS_SQL) >> (lambda rows:
                    put_line("[I] Preview victims_cached_enh (first 10):") ^
                    put_line("\n".join(
                        f"  art={r['article_id']} inc={r['incident_idx']} "
                        f"vic={r['victim_idx']} id={r['victim_row_id']} "
                        f"name='{r['victim_name_raw']}' age={r['victim_age']} "
                        f"sex={r['victim_sex']} race={r['victim_race']}"
                        for r in rows
                    ))
                )
            ),
            attach_sqlite_path=env["db_path"],
        ) ^
        put_line("[I] Done.") ^
        pure(NextStep.CONTINUE)
    )
