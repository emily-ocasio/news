# copilot-instructions.md


## 1. Project Overview

This project is a terminal-based tool for managing, analyzing, and interacting with news articles and crime data. The data is stored and managed in a SQLite database  newarticles.db and the logic includes managing calls to OpenAI GPT API. The main entry point is `runarticles.py`.

---

## 2. Architecture and Main Components

### 2.1 Overall structure:
- Program uses a monadic functional programming style, following patterns from PureScript and Haskell. In particular, it uses a Run monad to manage side effects.  Generic monadic abstracations are in the pymonad folder. Code here is inspired by the pymonad library but has been heavily modified, and follows more of the PureScript/Haskell style.

### 2.2. Starting Point
- The main entry point is `runarticles.py`. It establishes the main trampoline loop, where each iteration presents the main menu, and based on which selection, it invokes a corresponding controller.  All activity occurs within the context of a Run stack, using one or more of the eliminators in pymonad/run.py and pymonad/runsql.py, pymonad/runopenai.py, and pymonad/runsplink.py.

### 2.3. Controllers
- Controllers run within Run effect context and usually follow a pattern where one or more articles are retrieved, then processed in some way (for example submitting them to OpenAI for analysis).

### 2.4 Applicative Validation
- The program uses applicative validation to validate user input.  This is implemented in pymonad/validation.py for the definition of the V applicative and pymonad/validate_run.py for further helper functions to use within the run context.  The main idea is that multiple validation checks can be run in parallel, and all errors are collected and reported to the user at once, rather than failing fast on the first error.

### 2.4 Database
- Uses SQLite (`newarticles.db`) for persistent storage.
- Database schema and setup scripts are in `.sql` files- the main schema is in articles.sql.  runsql eliminator in `pymonad/runsql.py` is used to run SQL queries within the Run context.

### 2.5 OpenAI Integration
- Uses OpenAI GPT models for text generation and analysis.
- The runopenai eliminator in `pymonad/runopenai.py` is used to make API calls within the Run context.
- API keys are stored in `secr_apis/gpt3_key.py`.
- Calls to OpenAI use structured response using Pydantic models defined within state.py.
- The json schema for GPT extraction of incident details is in `gpt_extract_schema.json`. The corresponding json is stored in the database in the gptVictimJson column of the articles table.

### 2.6 Splink Integration
- Uses Splink for entity resolution and data linkage.
- Calls to Splink do not have their own eliminator but are resolved within pymonad/dispatch.py using the run_base_effect eliminator. Helper functions for Splink are in pymonad/runsplink.py.
- Data from the sqlite database is utilized by Splink through the DuckDB connector. Further caching of data trasnformations is stored within DuckDB tables.
- Extraction of data into DuckDB is handled primarily by the controller code in incidents_setup.py.
- API keys are stored in `secr_apis/splink_key.py`.

### 2.7 Geocoding Integration
- For articles refering to homicides in Washington DC, a call to the MAR 2 API is performed. There is no separate eliminator for this, but the call is made within pymonad/dispatch.py using the run_base_effect eliminator.


