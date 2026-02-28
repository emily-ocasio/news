"""
Set up DuckDB views for incident extraction from gptVictimJson (SQLite).
"""
import re

from pymonad import (
    Run,
    pure,
    put_line,
    ask,
    with_duckdb,
    sql_script,
    sql_query,
    sql_exec,
    SQL,
    Array,
    Environment,
    SQLParams,
    String,
    Unit,
    unit,
    EmbeddingModel,
    openai_embeddings,
)
from pymonad.traverse import array_traverse_run
from menuprompts import NextStep

SUMMARY_EMBED_MODEL = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
SUMMARY_EMBED_DIM = 1536
SUMMARY_EMBED_BATCH_SIZE = 100
ZERO_VEC_SQL = f"ARRAY[{', '.join('0' for _ in range(SUMMARY_EMBED_DIM))}]"


def _vec_to_array_sql(vec: tuple[float, ...]) -> str:
    return f"ARRAY[{','.join(format(x, '.17g') for x in vec)}]"


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _ensure_openai_embedding_cache_table() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """
            CREATE TABLE IF NOT EXISTS sbert_cache (
                input_text VARCHAR,
                vec DOUBLE[384]
            );
            """
            )
        )
        ^
        sql_exec(
            SQL(
                f"""
            CREATE TABLE IF NOT EXISTS openai_embedding_cache (
                model VARCHAR,
                dimensions INTEGER,
                input_text VARCHAR,
                vec DOUBLE[{SUMMARY_EMBED_DIM}]
            );
            """
            )
        )
        ^ sql_exec(
            SQL(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_openai_embedding_cache_key
            ON openai_embedding_cache(model, dimensions, input_text);
            """
            )
        )
    )


def _ensure_summary_vec_column() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                f"""
            ALTER TABLE incidents_cached
            ADD COLUMN IF NOT EXISTS summary_vec DOUBLE[{SUMMARY_EMBED_DIM}];
            """
            )
        )
        ^ sql_exec(
            SQL(
                f"""
            ALTER TABLE incidents_cached
            ALTER COLUMN summary_vec TYPE DOUBLE[{SUMMARY_EMBED_DIM}]
            USING CAST(NULL AS DOUBLE[{SUMMARY_EMBED_DIM}]);
            """
            )
        )
    )


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _embed_missing_texts(texts: list[str]) -> Run[list[tuple[str, tuple[float, ...]]]]:
    if len(texts) == 0:
        return pure([])

    batches = _chunked(texts, SUMMARY_EMBED_BATCH_SIZE)

    def _loop(idx: int, acc: list[tuple[str, tuple[float, ...]]]) -> Run[list[tuple[str, tuple[float, ...]]]]:
        if idx >= len(batches):
            return pure(acc)
        batch = batches[idx]
        return openai_embeddings(
            input_texts=batch,
            model=SUMMARY_EMBED_MODEL,
            dimensions=SUMMARY_EMBED_DIM,
            normalize=True,
        ) >> (
            lambda vecs: _loop(idx + 1, acc + list(zip(batch, vecs)))
        )

    return _loop(0, [])


def _cache_miss_vectors(texts: list[str]) -> Run[Unit]:
    if len(texts) == 0:
        return pure(unit)

    return _embed_missing_texts(texts) >> (
        lambda rows: array_traverse_run(
            Array.make(tuple(rows)),
            lambda rec: sql_exec(
                SQL(
                    f"""
                INSERT INTO openai_embedding_cache (model, dimensions, input_text, vec)
                VALUES (?, ?, ?, {_vec_to_array_sql(rec[1])})
                ON CONFLICT(model, dimensions, input_text) DO NOTHING;
                """
                ),
                SQLParams(
                    (
                        String(SUMMARY_EMBED_MODEL.value),
                        SUMMARY_EMBED_DIM,
                        String(rec[0]),
                    )
                ),
            ),
        ).map(lambda _: unit)
    )


def _update_single_summary_vec(row) -> Run[Unit]:
    key = str(row.get("summary_norm") or "").strip()
    article_id = row["article_id"]
    incident_idx = row["incident_idx"]

    if key == "":
        return sql_exec(
            SQL(
                """
            UPDATE incidents_cached
            SET summary_vec = NULL
            WHERE article_id = ? AND incident_idx = ?;
            """
            ),
            SQLParams((article_id, incident_idx)),
        )

    if key.lower() == "no details":
        return sql_exec(
            SQL(
                f"""
            UPDATE incidents_cached
            SET summary_vec = {ZERO_VEC_SQL}
            WHERE article_id = ? AND incident_idx = ?;
            """
            ),
            SQLParams((article_id, incident_idx)),
        )

    return sql_exec(
        SQL(
            """
        UPDATE incidents_cached
        SET summary_vec = (
            SELECT vec
            FROM openai_embedding_cache
            WHERE model = ?
              AND dimensions = ?
              AND input_text = ?
        )
        WHERE article_id = ? AND incident_idx = ?;
        """
        ),
        SQLParams(
            (
                String(SUMMARY_EMBED_MODEL.value),
                SUMMARY_EMBED_DIM,
                String(key),
                article_id,
                incident_idx,
            )
        ),
    )


def _update_summary_vectors_for_rows(env: Environment, rows: Array) -> Run[Unit]:
    """
    Populate summary vectors for a set of incidents using OpenAI embeddings.
    """
    _ = env
    row_list = list(rows)
    if len(row_list) == 0:
        return pure(unit)

    embed_keys = [
        str(row.get("summary_norm") or "").strip()
        for row in row_list
        if str(row.get("summary_norm") or "").strip() not in ("",)
        and str(row.get("summary_norm") or "").strip().lower() != "no details"
    ]
    unique_keys = list(dict.fromkeys(embed_keys))

    def _missing_keys(existing_rows: Array) -> Run[Unit]:
        existing = {str(r["input_text"]) for r in existing_rows}
        missing = [k for k in unique_keys if k not in existing]
        return (
            put_line(f"[I] OpenAI embedding cache hit={len(existing)} miss={len(missing)}")
            ^ _cache_miss_vectors(missing)
            ^ array_traverse_run(
                Array.make(tuple(row_list)),
                _update_single_summary_vec,
            ).map(lambda _: unit)
        )

    if len(unique_keys) == 0:
        return (
            _ensure_openai_embedding_cache_table()
            ^ _ensure_summary_vec_column()
            ^ array_traverse_run(
                Array.make(tuple(row_list)),
                _update_single_summary_vec,
            ).map(lambda _: unit)
        )

    values_sql = ",".join(f"('{_sql_quote(k)}')" for k in unique_keys)
    return (
        _ensure_openai_embedding_cache_table()
        ^ _ensure_summary_vec_column()
        ^ sql_query(
            SQL(
                f"""
            WITH keys(input_text) AS (
                VALUES {values_sql}
            )
            SELECT k.input_text
            FROM keys k
            JOIN openai_embedding_cache c
              ON c.input_text = k.input_text
             AND c.model = '{_sql_quote(SUMMARY_EMBED_MODEL.value)}'
             AND c.dimensions = {SUMMARY_EMBED_DIM};
            """
            )
        )
        >> _missing_keys
    )


def _row_update_run(env: Environment, row) -> Run[Unit]:
    """
    Compatibility wrapper kept for callers that still process a single row.
    """
    return _update_summary_vectors_for_rows(env, Array.make((row,)))


# --- DuckDB SQL ---
# --- Victim-level extraction & feature engineering ---

# 1) Explode victim array -> one row per victim mention
CREATE_VICTIMS_CACHED_SQL = SQL(
    """--sql
CREATE OR REPLACE TABLE victims_cached AS
SELECT
  a.article_id,
  a.city_id,
  a.publish_date,
  a.year, a.month, a.day,
  a.incident_idx,
  a.summary_vec,
  CAST(v.key AS INTEGER) AS victim_idx,
  json_extract_string(v.value, '$.victim_name')      AS victim_name_raw,
  try_cast(json_extract_string(v.value, '$.victim_age') AS INTEGER) AS victim_age_raw,
  json_extract_string(v.value, '$.victim_sex')       AS victim_sex,
  json_extract_string(v.value, '$.victim_race')      AS victim_race,
  json_extract_string(v.value, '$.victim_ethnicity') AS victim_ethnicity,
  json_extract_string(v.value, '$.relationship') AS victim_relationship,
  json_extract_string(v.value, '$.relationship') AS relationship,

  a.incident_date, a.event_start_day, a.event_end_day,
  a.weapon, a.circumstance,

  -- keep raw offender_name so we can normalize it with the same logic as victims
  a.offender_name,
  a.offender_count,

  coalesce(
    try_cast(json_extract_string(v.value, '$.victim_count') AS INTEGER),
    a.victim_count
  ) AS victim_count,

  concat_ws(':', CAST(a.article_id AS VARCHAR), CAST(a.incident_idx AS VARCHAR), CAST(CAST(v.key AS INTEGER) AS VARCHAR)) AS victim_row_id
FROM incidents_cached a
CROSS JOIN json_each(CAST(a.incident_json AS JSON), '$.victim') AS v
WHERE json_type(CAST(a.incident_json AS JSON), '$.victim') = 'ARRAY';
"""
)

# 2) Normalize names + sanitize age,
# add coarse age bucket + parsed forename/surname
CREATE_VICTIMS_CACHED_ENH_SQL = SQL(
    r"""--sql
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
    regexp_replace(name_clean, '^(?i)(mr|mrs|ms|miss|dr|rev|ofc|officer)\s+', '', 'g') 
      AS name_no_title,
    regexp_replace(offender_name_clean, 
      '^(?i)(mr|mrs|ms|miss|dr|rev|ofc|officer)\s+', '', 'g') AS offender_no_title
  FROM norm
),
strip2 AS (
  SELECT
    *,
    regexp_replace(name_no_title, '\s+(?i)(jr|sr|ii|iii|iv)\.?$', '', 'g') AS name_core,
    regexp_replace(offender_no_title, '\s+(?i)(jr|sr|ii|iii|iv)\.?$', '', 'g')
      AS offender_name_core
  FROM strip
),
parts AS (
  SELECT
    *,
    CASE
      WHEN victim_name_raw IS NULL THEN NULL
      WHEN name_core LIKE '% %' THEN
        CASE
          -- If leading initial + one name, keep initial as forename (e.g., "A Gonzalez")
          WHEN regexp_matches(name_core, '^[a-z]\s+[a-z''\-]+$')
            THEN regexp_extract(name_core, '^([a-z])', 1)
          -- If leading initial + multiple names, use the first name after initial (e.g., "A Orlando Gonzalez")
          WHEN regexp_matches(name_core, '^[a-z]\s+[a-z''\-]+\s+')
            THEN regexp_extract(name_core, '^[a-z]\s+([a-z''\-]+)', 1)
          ELSE regexp_extract(name_core, '^([a-z''\-]+)', 1)
        END
      ELSE NULL
    END AS victim_forename_norm,
    CASE
      WHEN victim_name_raw IS NULL THEN NULL
      WHEN name_core LIKE '% %' THEN
        CASE
          WHEN regexp_matches(name_core, 
              '(?i)(?:^|\s)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+$')
            THEN regexp_extract(name_core, 
                '((?i)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+)$', 1)
          ELSE regexp_extract(name_core, '([a-z''\-]+)$', 1)
        END
      ELSE name_core
    END AS victim_surname_norm,
    CASE
      WHEN offender_name IS NULL THEN NULL
      ELSE regexp_extract(offender_name_core, '^([a-z''\-]+)', 1)
    END AS offender_forename_norm,
    CASE
      WHEN offender_name IS NULL THEN NULL
      WHEN offender_name_core LIKE '% %' THEN
        CASE
          WHEN regexp_matches(offender_name_core, 
            '(?i)(?:^|\s)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+$')
            THEN regexp_extract(offender_name_core, 
              '((?i)(?:de la|del|de|van|von|da|di|la|le|st)\s+[a-z''\-]+)$', 1)
          ELSE regexp_extract(offender_name_core, '([a-z''\-]+)$', 1)
        END
      ELSE NULL
    END AS offender_surname_norm
  FROM strip2
)
SELECT
  p.*,

  -- Concatenations
  CASE
    WHEN victim_forename_norm IS NOT NULL AND victim_surname_norm IS NOT NULL
      THEN victim_forename_norm || ' ' || victim_surname_norm
    WHEN victim_forename_norm IS NOT NULL
      THEN victim_forename_norm
    WHEN victim_surname_norm IS NOT NULL
      THEN victim_surname_norm
  END AS victim_fullname_concat,
  CASE
    WHEN offender_forename_norm IS NOT NULL AND offender_surname_norm IS NOT NULL
      THEN offender_forename_norm || ' ' || offender_surname_norm
    WHEN offender_forename_norm IS NOT NULL
      THEN offender_forename_norm
    WHEN offender_surname_norm IS NOT NULL
      THEN offender_surname_norm
  END AS offender_fullname_concat,

  -- Individual-part Soundex (from splink_udfs)
  CASE WHEN victim_forename_norm IS NOT NULL
    THEN soundex(victim_forename_norm)
  END AS victim_forename_soundex,
  CASE WHEN victim_surname_norm IS NOT NULL
    THEN soundex(victim_surname_norm)
  END AS victim_surname_soundex,
  CASE WHEN offender_forename_norm IS NOT NULL
    THEN soundex(offender_forename_norm)
  END AS offender_forename_soundex,
  CASE WHEN offender_surname_norm IS NOT NULL
    THEN soundex(offender_surname_norm)
  END AS offender_surname_soundex,

  -- Normalized full-name strings (letters+spaces only)
  regexp_replace(name_core, '[^a-z ]', '', 'g') AS victim_name_norm,
  regexp_replace(offender_name_core, '[^a-z ]', '', 'g') AS offender_name_norm,

  -- Age cleanup + buckets
  CASE WHEN victim_age_raw BETWEEN 0 AND 120 THEN victim_age_raw END AS victim_age,
  CASE WHEN victim_age_raw BETWEEN 0 AND 120 THEN floor(victim_age_raw/5) END
    AS victim_age_bucket5,

  -- Offender age from incidents
  i.offender_age,

  -- Offender count from incidents
  i.offender_count,

  -- Offender sex from incidents
  i.offender_sex,

  -- Offender race from incidents
  i.offender_race,

  -- Offender ethnicity from incidents
  i.offender_ethnicity,

  -- Date fields from incidents_cached
  i.date_precision,
  i.midpoint_day,

  -- Unify article ids for blocking/linkage (single-id CSV)
  CAST(p.article_id AS varchar) AS article_ids_csv
FROM parts p
LEFT JOIN incidents_cached i
  ON p.article_id = i.article_id
 AND p.incident_idx = i.incident_idx;
"""
)

COUNT_VICTIMS_SQL = SQL("SELECT COUNT(*) AS n FROM victims_cached_enh;")
PREVIEW_VICTIMS_SQL = SQL(
    """SELECT article_id, incident_idx, victim_idx, victim_row_id,
       victim_name_raw, victim_age, victim_sex, victim_race, victim_ethnicity,
       victim_relationship
FROM victims_cached_enh
LIMIT 10;
"""
)


def _extract_create_table_select_sql(create_sql: SQL, table_name: str) -> SQL:
    """
    Extract the SELECT/WITH query body from:
      CREATE OR REPLACE TABLE <table_name> AS <query>;
    """
    pattern = (
        rf"CREATE\s+OR\s+REPLACE\s+TABLE\s+{re.escape(table_name)}\s+AS\s*(.*)\s*;\s*$"
    )
    m = re.search(pattern, str(create_sql), flags=re.IGNORECASE | re.DOTALL)
    if m is None:
        raise ValueError(f"Could not extract SELECT SQL for table: {table_name}")
    return SQL(m.group(1))


VICTIMS_CACHED_SELECT_SQL = _extract_create_table_select_sql(
    CREATE_VICTIMS_CACHED_SQL,
    "victims_cached",
)
VICTIMS_CACHED_ENH_SELECT_SQL = _extract_create_table_select_sql(
    CREATE_VICTIMS_CACHED_ENH_SQL,
    "victims_cached_enh",
)
# --- Incident-level extraction & feature engineering ---
CREATE_STG_ARTICLE_INCIDENTS_SQL = SQL(
    """--sql
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
    CASE WHEN COALESCE(json_extract_string(i.value,'$.killing_method'), json_extract_string(i.value,'$.weapon')) = 'beating' THEN 'personal weapon' ELSE COALESCE(json_extract_string(i.value,'$.killing_method'), json_extract_string(i.value,'$.weapon')) END AS weapon,
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
    trim(regexp_replace(regexp_replace(coalesce(summary, ''), '[[:cntrl:]]+', '', 'g'), '[[:space:]]+', ' ', 'g')) AS summary_norm
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
FROM norm
WHERE year_raw >= 1977;
"""
)

CREATE_INCIDENTS_CACHED_SQL = SQL(
    """
CREATE OR REPLACE TABLE incidents_cached AS
SELECT * FROM stg_article_incidents;
"""
)

COUNT_VIEW_SQL = SQL("SELECT COUNT(*) AS n FROM stg_article_incidents;")
COUNT_CACHE_SQL = SQL("SELECT COUNT(*) AS n FROM incidents_cached;")
PREVIEW_CACHE_SQL = SQL(
    """
SELECT article_id, incident_idx, year, month, day, incident_date,
       event_start_day, event_end_day, weapon, circumstance,
       substr(summary, 1, 80) AS summary_snip
FROM incidents_cached
LIMIT 10;
"""
)


def build_incident_views() -> Run[NextStep]:
    """
    Steps:
      1) In SQLite (already under run_sql): rebuild filtered subset table.
      2) In DuckDB: attach SQLite, rebuild view from subset, materialize cache.
      3) Print counts and preview. Return to menu.
    Requires env:
      Uses existing SQLite + DuckDB connections from the environment.
    """
    return ask() >> (
        lambda env:
        # --- 1) Rebuild subset in SQLite
        # (this runs in the existing run_sql context) ---
        # For now, just articles labeled 'M' in CLASS_WP
        # which represents the articles from Washington Post
        # in future, we will include other newspapers
        # and set up city_id appropriately
        put_line("[I] Rebuilding SQLite subset table articles_wp_subset…")
        ^ sql_exec(SQL("DROP TABLE IF EXISTS articles_wp_subset;"))
        ^ sql_exec(
            SQL(
                """
            CREATE TABLE articles_wp_subset AS
            SELECT RecordId, Publication, PubDate, Title, FullText, gptVictimJson, LastUpdated
            FROM articles
            WHERE Dataset='CLASS_WP' AND gptClass='M';
        """
            )
        )
        ^ sql_exec(
            SQL(
                "CREATE INDEX IF NOT EXISTS idx_articles_wp_subset_pk "
                "ON articles_wp_subset(RecordId);"
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM articles_wp_subset;"))
        >> (lambda rows: put_line(f"[I] SQLite subset rows: {rows[0]['n']}"))
        ^
        # --- 2) Build DuckDB view + materialized cache from the subset ---
        put_line("[I] Building DuckDB view and materialized cache…")
        ^ with_duckdb(
            (
                sql_script(
                    SQL(
                        """
                    INSTALL splink_udfs FROM community;
                    LOAD splink_udfs;
                """
                    )
                )
                ^ sql_script(CREATE_STG_ARTICLE_INCIDENTS_SQL)
                ^ sql_query(COUNT_VIEW_SQL)
                >> (lambda rows: put_line(f"[I] View rows: {rows[0]['n']}"))
                ^ sql_script(CREATE_INCIDENTS_CACHED_SQL)
                ^ sql_query(
                    SQL(
                        """
                SELECT article_id, incident_idx, summary_norm
                FROM incidents_cached;
                """
                    )
                )
                >> (
                    lambda rows: put_line(
                        f"[I] Retrieved {len(rows)} rows for vectorization…"
                    )
                    ^ _update_summary_vectors_for_rows(env, Array.make(tuple(rows)))
                )
                ^ put_line("[I] Updated vectorized summary for all incidents.")
                ^ sql_script(CREATE_VICTIMS_CACHED_SQL)
                ^ sql_script(CREATE_VICTIMS_CACHED_ENH_SQL)
                ^ sql_query(COUNT_VICTIMS_SQL)
                >> (
                    lambda rows: put_line(
                        f"[I] victims_cached_enh rows: {rows[0]['n']}"
                    )
                )
                ^ sql_query(PREVIEW_VICTIMS_SQL)
                >> (
                    lambda rows: put_line("[I] Preview victims_cached_enh (first 10):")
                    ^ put_line(
                        "\n".join(
                            f"  art={r['article_id']} inc={r['incident_idx']} "
                            f"vic={r['victim_idx']} id={r['victim_row_id']} "
                            f"name='{r['victim_name_raw']}' age={r['victim_age']} "
                            f"sex={r['victim_sex']} race={r['victim_race']}"
                            for r in rows
                        )
                    )
                )
            ),
        )
        ^ put_line("[I] Done.")
        ^ pure(NextStep.CONTINUE)
    )
