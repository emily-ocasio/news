# copilot-instructions.md
Start every chat response to questions about this program with the following text: "THANK YOU FOR THE QUESTION ABOUT THE NEWS ARTICLES ANALYSIS PROJECT."

## Mypy (dmypy) Type Checking

After any code changes:

1. Run the mypy daemon using the same Python interpreter and flags the extension uses:

/Users/wendell/miniforge3/envs/news/bin/dmypy
--status-file .dmypy.json 
run 
-- 
--python-executable /Users/wendell/miniforge3/envs/news/bin/python 
. 
--show-error-end 
--no-error-summary 
--no-pretty 
--no-color-output 
--config-file mypy.ini

2. If any type errors are reported (non-zero exit status or output with errors), treat them as failures and fix them before completing the change.

3. Optionally check daemon status:
/Users/wendell/miniforge3/envs/news/bin/python -m mypy.dmypy 
--status-file .dmypy.json status

## Pyright (Pylance) Type Checking

After any code changes:

1. Run Pyright using the CLI-specific config (keeps VS Code Pylance behavior unchanged):

/Users/wendell/miniforge3/envs/news/bin/pyright --project pyrightconfig.cli.json

2. If any type errors are reported (non-zero exit status or output with errors), treat them as failures and fix them before completing the change.


## 1. Project Overview

This project is a terminal-based tool for managing, analyzing, and interacting with news articles and crime data. The data is stored and managed in a SQLite database  newarticles.db and the logic includes managing calls to OpenAI GPT API. The main entry point is [runarticles.py](../runarticles.py).

---

## 2. Architecture and Main Components

### 2.1 Overall structure:
- Program uses a monadic functional programming style, following patterns from PureScript and Haskell. In particular, it uses a Run monad to manage side effects.  Generic monadic abstracations are in the pymonad folder. Code here is inspired by the pymonad library but has been heavily modified, and follows more of the PureScript/Haskell style.

### 2.2. Starting Point
- The main entry point is [runarticles.py](../runarticles.py). It establishes the main trampoline loop, where each iteration presents the main menu, and based on which selection, it invokes a corresponding controller.  All activity occurs within the context of a Run stack, using one or more of the eliminators in [pymonad/run.py](../pymonad/run.py), [pymonad/runsql.py](../pymonad/runsql.py), [pymonad/runopenai.py](../pymonad/runopenai.py), [pymonad/runsplink.py](../pymonad/runsplink.py), and [pymonad/dispatch.py](../pymonad/dispatch.py).

### 2.3. Controllers
- Controllers run within Run effect context and usually follow a pattern where one or more articles are retrieved, then processed in some way (for example submitting them to OpenAI for analysis).

### 2.3.1 Menu Options and Controllers
- The main menu in [mainmenu.py](../mainmenu.py) dispatches to specific controllers based on user choice:
  - GPT (S): `second_filter()` in [gpt_filtering.py](../gpt_filtering.py) - Uses GPT to classify articles as homicide-related or not, saving classifications. The classification is stored in the gptClass column of the articles table as a coded value. "M_PRELIM" means that the article is filtered as a potential homicide article after the second GPT filter. GPT classes starting with "SP_" mean that this is a special case where the validator found a term that means do not use GPT for classsification. This special case is often used for articles of specific homicides that are so notable that they resulti in a large number of articles, so we want to include them without GPT review.
  - FIX (F): `fix_article()` in [fixarticle.py](../fixarticle.py) - Allows reviewing and re-processing a single article by ID.
  - EXTRACTION (G): `gpt_incidents()` in [incidents.py](../incidents.py) - Extracts detailed incident info from articles using GPT, saving JSON.
  - INCIDENTS (I): `build_incident_views()` in [incidents_setup.py](../incidents_setup.py) - Builds SQLite/DuckDB views for incident data preparation for Splink.
  - GEOCODE (M): `geocode_incidents()` in [geocode_incidents.py](../geocode_incidents.py) - Geocodes DC homicide addresses via MAR API.
  - DEDUP (D): `dedupe_incidents()` in [incidents_dedupe.py](../incidents_dedupe.py) - Deduplicates incidents using Splink.
  - UNNAMED (U): `match_unnamed_victims()` in [unnamed_match.py](../unnamed_match.py) - Matches unnamed victims using Splink.
  - LINK (L): `match_article_to_shr_victims()` in [shr_match.py](../shr_match.py) - Links articles to SHR victims using Splink.
  - SPECIAL (P): `review_special_cases()` in [special_case_review.py](../special_case_review.py) - Reviews special case articles.
  - SPECIAL_ADD (Y): `add_special_articles()` in [special_add.py](../special_add.py) - Adds special case articles to the entities.
  - Other options (REVIEW, NEW, etc.) are placeholders that log and continue to menu.

### 2.3.2 Run Effects Usage
- Controllers compose Run effects for side-effect management:
  - Database operations (queries/executes) via [pymonad/runsql.py](../pymonad/runsql.py) eliminator.
  - GPT API calls via [pymonad/runopenai.py](../pymonad/runopenai.py) eliminator.
  - Splink entity resolution and geocoding via `run_base_effect` in [pymonad/dispatch.py](../pymonad/dispatch.py).
  - IO (input/output) and state via core Run monad in [pymonad/run.py](../pymonad/run.py).
- Effects are stacked in Run contexts, allowing pure functional composition with controlled side effects.

### 2.4 Applicative Validation
- The program uses applicative validation to validate user input.  This is implemented in pymonad/validation.py for the definition of the V applicative and pymonad/validate_run.py for further helper functions to use within the run context.  The main idea is that multiple validation checks can be run in parallel, and all errors are collected and reported to the user at once, rather than failing fast on the first error.

### 2.4 Database
- Uses SQLite (`newarticles.db`) for persistent storage.
- Database schema and setup scripts are in `.sql` files- the main schema is in articles.sql.  runsql eliminator in [pymonad/runsql.py](../pymonad/runsql.py) is used to run SQL queries within the Run context.

### 2.5 OpenAI Integration
- Uses OpenAI GPT models for text generation and analysis.
- The runopenai eliminator in [pymonad/runopenai.py](../pymonad/runopenai.py) is used to make API calls within the Run context.
- API keys are stored in [secr_apis/gpt3_key.py](../secr_apis/gpt3_key.py).
- Calls to OpenAI use structured response using Pydantic models defined within state.py.
- The json schema for GPT extraction of incident details is in [gpt_extract_schema.json](../gpt_extract_schema.json). The corresponding json is stored in the database in the gptVictimJson column of the articles table.

### 2.6 Splink Integration
- Uses Splink for entity resolution and data linkage.
- Calls to Splink are done via the run_splink eliminator in [pymonad/runsplink.py](../pymonad/runsplink.py).
- Data from the sqlite database is utilized by Splink through the DuckDB connector. Further caching of data trasnformations is stored within DuckDB tables.
- Extraction of data from the GPT structured output in JSON format stored in sqlite into DuckDB to stage for splink is handled primarily by the controller code in incidents_setup.py.
- Detailed information about splink documentation is available at ./splink-reference.md
- Splink library source code is in /Users/wendell/miniforge3/envs/news/lib/python3.13/site-packages/splink


### 2.7 Geocoding Integration
- For articles refering to homicides in Washington DC, a call to the MAR 2 API is performed. There is no separate eliminator for this, but the call is made within pymonad/dispatch.py using the run_base_effect eliminator.

### 2.8 High-Level User Workflow Steps
The overall goal of the program is to analyze the text of thousands of newspaper articles stored in the SQLite database to determine whether the articles show victims of homicides in the correct geographic area. As the code is currently written, there is an assumption that a preliminary "first filter" was already done using basic regex. Then this is followed by a "second filter" using GPT API. This second filter determines more precisely whether the articles are to be included in the analysis because they refer to homicides in the correct geographic area. Then for those articles that are still considered homicide, then another pass is done to extract the specific structured data, also using GPT API -- this data is stored in the SQLite database in a text json string for each article. Next step is to setup the data from this json in SQLite into DuckDB tables/views for subsequent use by Splink. Then the MAR API is used to augment the data based on the text description of the location into a precise longitude/latitude. Then an initial Splink deduplication using clustering is done to create a list of entities where each entity corresponds to a single victim and includes information from multiple articles. This initial deduplication only includes victims where the article has named the victims. Articles where the victims do not have names are referred as orphans and they are then matched to entities using a second Splink pass, this one being a linkage instead of clustering.

The typical user workflow for analyzing news articles and crime data follows a sequential process, starting from raw articles and progressing through classification, extraction, geocoding, deduplication, and linkage. Each step is invoked via the main menu in [runarticles.py](../runarticles.py) and involves controllers that operate within Run monad contexts for side-effect management.

1. **Preliminary First Filter (Assumed Done)**: A basic regex filter to identify potential homicide-related articles.
2. **Initial Article Classification (Second Filter)**: Select GPT (S) to classify articles as homicide-related using GPT-4 via `second_filter()` in [gpt_filtering.py](../gpt_filtering.py). This filters the dataset and saves classifications.
3. **Detailed Incident Extraction**: Select EXTRACTION (G) to extract detailed incident info (e.g., victim details) from classified articles using GPT via `gpt_incidents()` in [incidents.py](../incidents.py), saving structured JSON to the `gptVictimJson` column.
4. **Incident Data Preparation**: Select INCIDENTS (I) to build SQLite/DuckDB views for incident data preparation for Splink using `build_incident_views()` in [incidents_setup.py](../incidents_setup.py).
5. **Geocoding**: Select GEOCODE (M) to geocode DC homicide addresses via the MAR 2 API using `geocode_incidents()` in [geocode_incidents.py](../geocode_incidents.py).
6. **Incident Deduplication**: Select DEDUP (D) to deduplicate incidents using Splink via `dedupe_incidents()` in [incidents_dedupe.py](../incidents_dedupe.py).
7. **Unnamed Victim Matching**: Select UNNAMED (U) to match unnamed victims to existing deduped clusters using Splink via `match_unnamed_victims()` in [unnamed_match.py](../unnamed_match.py).
8. **Linking to SHR Data**: Select LINK (L) to set up SHR data in DuckDB from the SQLite shr table and link article victims to SHR victims using Splink via `match_article_to_shr_victims()` in [shr_match.py](../shr_match.py).
9. **Optional Fixes/Reviews**: Use FIX (F) for manual review/re-processing of individual articles, or placeholders like REVIEW/NEW for additional tasks.

This workflow ensures data integrity through monadic composition, with all operations logged and validated via applicative validation.
