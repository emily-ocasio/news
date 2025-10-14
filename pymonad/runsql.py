"""
Intent and eliminator for SQL (database) effects.
"""

# pylint:disable=W0212
from dataclasses import dataclass
from typing import TypeVar
from typing import Any, cast

import sqlite3
import pandas as pd
import duckdb
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import FormulaRule
from openpyxl.cell import Cell

from .array import Array
from .run import Run, _unhandled, ask
from .string import String


class SQL(String):
    """
    Represents a SQL string.
    """


type SQLParam = String | int | float | None


class SQLParams(Array[SQLParam]):
    """
    Represents a list of SQL parameters.
    """

    def to_params(self) -> tuple:
        """
        Convert SQLParams to a tuple of parameters.
        """

        def _convert(param: SQLParam) -> str | int | float | None:
            match param:
                case String(s):
                    return s
                case _:
                    return param

        return tuple(_convert(param) for param in self.a)


class SQLExecutionError(Exception):
    """
    Raised by run_sqlite/run_duckdb performer when the underlying
    driver raises an exception. Captures SQL and params.
    """

    def __init__(self, sql: str, params: tuple | None, original: Exception):
        self.sql = sql
        self.params = params
        self.original = original
        super().__init__(str(original))

    def __str__(self) -> str:
        p = "None" if self.params is None else repr(self.params)
        return (
            f"{self.original.__class__.__name__}: {self.original}\n"
            f"--- SQL Context --------------------------------\n"
            f"SQL:\n{self.sql}\n"
            f"Params: {p}\n"
            f"------------------------------------------------"
        )


# --- DB intents ---
@dataclass(frozen=True)
class SqlQuery:
    """
    Represents a SQL query with parameters.
    """

    sql: SQL
    params: SQLParams = SQLParams(())


@dataclass(frozen=True)
class SqlExec:
    """
    Represents a SQL execution command with parameters.
    """

    sql: SQL
    params: SQLParams = SQLParams(())


@dataclass(frozen=True)
class SqlScript:
    """
    Represents a SQL script with multiple statements.
    """

    sql: SQL  # multi-statement DDL/DML


@dataclass(frozen=True)
class InTransaction:
    """
    Represents a database transaction.
    """

    program: Run[Any]  # sub-program run atomically


@dataclass(frozen=True)
class SqlExport:
    """
    Intent to export results of a SQL query into a spreadsheet (or fallback CSV),
    optionally with row-band formatting by group.
    """
    sql: SQL
    filename: str
    sheet: str | None = None
    band_by_group_col: str | None = None
    band_wrap: int = 2  # how many alternating bands/colors


def sql_query(sql: SQL, params: SQLParams = SQLParams(())) -> Run[Array]:
    """
    Smart constructor for SQL queries with intent.
    """
    return Run(lambda self: self._perform(SqlQuery(sql, params), self), _unhandled)


def sql_exec(sql: SQL, params: SQLParams = SQLParams(())) -> Run[None]:
    """
    Smart constructor for SQL execution with intent.
    """
    return Run(lambda self: self._perform(SqlExec(sql, params), self), _unhandled)


def sql_script(sql: SQL) -> "Run[None]":
    """
    Smart constructor for SQL scripts with intent.
    """
    return Run(lambda self: self._perform(SqlScript(sql), self), _unhandled)


def in_transaction(prog: "Run[A]") -> "Run[A]":
    """
    Smart constructor for database transactions with intent.
    """
    return Run(lambda self: self._perform(InTransaction(prog), self), _unhandled)


def sql_export(
    sql: SQL,
    filename: str,
    sheet: str | None = None,
    band_by_group_col: str | None = None,
    band_wrap: int = 2,
) -> Run[None]:
    """
    Smart constructor for exporting SQL query results to a spreadsheet (or CSV).
    """
    return Run(
        lambda self: self._perform(
            SqlExport(sql, filename, sheet, band_by_group_col, band_wrap),
            self,
        ),
        _unhandled,
    )


A = TypeVar("A")

def _apply_band_formatting_xlsx(
    filename: str, sheet: str, group_col: str, band_wrap: int = 2
) -> None:
    """
    After writing the Excel file, color rows in bands.
    Generic behavior:
      - If `group_col` already contains integer band indices in 0..band_wrap-1,
        apply conditional formats directly from that column (no helper col).
      - Otherwise, insert a helper column that alternates a band index whenever
        the `group_col` value changes (classic grouping bands).
    """
    try:
        wb = load_workbook(filename)
    except KeyError as e:
        print(f"[WARN] Cannot load {filename} for band formatting: {e}")
        return
    if sheet not in wb.sheetnames:
        wb.close()
        return
    ws = wb[sheet]

    # Read header row
    header = [cell.value for cell in ws[1]]
    if group_col not in header:
        wb.close()
        return

    # locate the (1-based) column index for the requested group_col
    grp_idx = header.index(group_col) + 1

    # Try DIRECT MODE: if the band column already contains explicit integers
    # within [0, band_wrap), use them as-is and apply per-index formulas.
    try:
        grp_col_letter_direct = cast(Cell, ws.cell(row=1, column=grp_idx)).column_letter
        sample_rows = min(ws.max_row, 101)
        seen_any = False
        direct_ok = True
        for row in range(2, sample_rows + 1):
            val = ws.cell(row=row, column=grp_idx).value
            if val is None or val == "":
                continue
            seen_any = True
            try:
                ival = int(str(val))
                if ival < 0 or ival >= band_wrap:
                    direct_ok = False
                    break
            except Exception:
                direct_ok = False
                break
        if seen_any and direct_ok:
            last_col = ws.max_column
            last_col_letter = cast(Cell, ws.cell(row=1, column=last_col)).column_letter
            data_range = f"A2:{last_col_letter}{ws.max_row}"
            # simple 3-color palette; cycles up to band_wrap
            fill_list = [
                PatternFill(start_color="FFEEEE", end_color="FFEEEE", fill_type="solid"),
                PatternFill(start_color="EEFFEE", end_color="EEFFEE", fill_type="solid"),
                PatternFill(start_color="EEEEFF", end_color="EEEEFF", fill_type="solid"),
            ]
            max_wrap = min(band_wrap, len(fill_list))
            for i in range(max_wrap):
                formula = f'${grp_col_letter_direct}2 = {i}'
                ws.conditional_formatting.add(
                    data_range,
                    FormulaRule(formula=[formula], fill=fill_list[i]),
                )
            wb.save(filename)
            wb.close()
            return
    except Exception:
        # fall through to helper-based mode
        pass

    # HELPER MODE: Insert helper column A with a rolling band counter that
    # increments when group_col value changes (alternating bands by group).
    ws.insert_cols(1)
    ws.cell(row=1, column=1, value="_band_helper")

    # After insert, group_col shifts right by one
    grp_col_letter = cast(Cell, ws.cell(row=1, column=(grp_idx + 1))).column_letter
    helper_letter = cast(Cell, ws.cell(row=1, column=1)).column_letter

    # Fill helper formulas
    for row in range(2, ws.max_row + 1):
        cell = cast(Cell, ws.cell(row=row, column=1))
        formula = (
            f"=MOD(IF(${grp_col_letter}{row} = ${grp_col_letter}{row-1}, "
            f"${helper_letter}{row-1}, ${helper_letter}{row-1} + 1), {band_wrap})"
        ) if row > 2 else "=0"
        cell.value = formula

    # Pattern fills
    fill_list = [
        PatternFill(start_color="FFEEEE", end_color="FFEEEE", fill_type="solid"),
        PatternFill(start_color="EEFFEE", end_color="EEFFEE", fill_type="solid"),
        PatternFill(start_color="EEEEFF", end_color="EEEEFF", fill_type="solid"),
    ]
    max_wrap = min(band_wrap, len(fill_list))

    # Apply conditional formatting across all columns A -> last column
    last_col = ws.max_column
    last_col_letter = cast(Cell, ws.cell(row=1, column=last_col)).column_letter
    for i in range(max_wrap):
        formula = f"${helper_letter}2 = {i}"
        ws.conditional_formatting.add(
            f"A2:{last_col_letter}{ws.max_row}",
            FormulaRule(formula=[formula], fill=fill_list[i]),
        )

    wb.save(filename)
    wb.close()

def run_sqlite(
    db_path: str,
    prog: Run[A],
    *,
    pragmas: dict[str, Any] | None = None,
    row_factory: Any = sqlite3.Row,  # name-based access: row["col"]
) -> Run[A]:
    """
    Eliminator for SQLite database calls.
    """

    def step(self_run: Run[Any]) -> A:
        parent = self_run._perform
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = row_factory  # rows act like dicts by column name
            if pragmas:
                with con:
                    for k, v in pragmas.items():
                        con.execute(f"PRAGMA {k}={v};")

            def perform(intent: Any, current: "Run[Any]") -> Any:
                match intent:
                    case SqlQuery(sql, params):
                        try:
                            cur = con.execute(str(sql), params.to_params())
                            rows = cur.fetchall()
                            cur.close()
                            return rows
                        except Exception as ex:  # noqa: BLE001
                            raise SQLExecutionError(
                                str(sql), params.to_params(), ex
                            ) from ex
                    case SqlExec(sql, params):
                        try:
                            con.execute(str(sql), params.to_params())
                            con.commit()
                            return None
                        except Exception as ex:  # noqa: BLE001
                            raise SQLExecutionError(
                                str(sql), params.to_params(), ex
                            ) from ex
                    case SqlScript(sql):
                        try:
                            con.executescript(str(sql))
                            return None
                        except Exception as ex:  # noqa: BLE001
                            raise SQLExecutionError(str(sql), None, ex) from ex

                    case SqlExport(sql, filename, sheet, band_by_group_col, band_wrap):
                        try:
                            df = pd.read_sql_query(str(sql), con)
                        except Exception as ex:
                            raise SQLExecutionError(str(sql), None, ex) from ex

                        try:
                            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                                df.to_excel(
                                    writer, sheet_name=sheet or "Sheet1", index=False
                                )
                        except (
                            PermissionError,
                            FileNotFoundError,
                            OSError,
                            ValueError,
                        ):
                            fallback = (
                                filename
                                if filename.lower().endswith(".csv")
                                else (filename + ".csv")
                            )
                            df.to_csv(fallback, index=False)

                        if band_by_group_col and filename.lower().endswith(".xlsx"):
                            _apply_band_formatting_xlsx(filename, sheet or "Sheet1",
                                                            band_by_group_col, band_wrap)
                        return None
                    case InTransaction(subprog):
                        try:
                            result = subprog._step(current)
                            con.commit()
                            return result
                        except Exception:
                            con.rollback()
                            raise

                    case _:
                        return parent(intent, current)

            inner = Run(prog._step, perform)
            return inner._step(inner)
        finally:
            con.close()

    return Run(step, lambda i, c: c._perform(i, c))


def run_duckdb(
    db_path: str,
    prog: Run[A],
    *,
    attach_sqlite_path: str | None = None,
    setup_sql: str | None = None,  # e.g., CREATE VIEWs, INSTALL/LOAD, etc.
) -> Run[A]:
    """
    Eliminator for DuckDB database calls.
    - Optionally ATTACH an existing SQLite DB via sqlite_scanner.
    - Optionally run a setup SQL script (CREATE VIEWs, INSTALL/LOAD, etc.)
    Normalizes query results to list of dicts (column-name access)
    to match how you currently use sqlite3.Row.
    """

    def step(self_run: Run[Any]) -> A:
        parent = self_run._perform
        con = duckdb.connect(db_path)
        try:
            # Optional: attach SQLite
            if attach_sqlite_path:
                con.execute("INSTALL sqlite_scanner;")
                con.execute("LOAD sqlite_scanner;")
                con.execute(f"ATTACH '{attach_sqlite_path}' AS sqldb (TYPE SQLITE);")

            # Optional: any one-off setup script (views, extensions, etc.)
            if setup_sql:
                con.execute(setup_sql)

            def _query_dicts(sql_s: str, params: SQLParams) -> list[dict]:
                try:
                    cur = con.execute(sql_s, params.to_params())
                    cols = [d[0] for d in (cur.description or [])]
                    rows = cur.fetchall()
                    return [dict(zip(cols, tup)) for tup in rows]
                except Exception as ex:  # noqa: BLE001
                    raise SQLExecutionError(sql_s, params.to_params(), ex) from ex

            def perform(intent: Any, current: "Run[Any]") -> Any:
                match intent:
                    case SqlQuery(sql, params):
                        return _query_dicts(str(sql), params)
                    case SqlExec(sql, params):
                        try:
                            con.execute(str(sql), params.to_params())
                            return None
                        except Exception as ex:  # noqa: BLE001
                            raise SQLExecutionError(
                                str(sql), params.to_params(), ex
                            ) from ex
                    case SqlScript(sql):
                        try:
                            con.execute(str(sql))
                            return None
                        except Exception as ex:  # noqa: BLE001
                            raise SQLExecutionError(str(sql), None, ex) from ex

                    case SqlExport(sql, filename, sheet, band_by_group_col, band_wrap):
                        # Try native DuckDB Excel export first
                        try:
                            raise duckdb.BinderException
                            con.execute("INSTALL excel;")
                            con.execute("LOAD excel;")
                            stmt = f"COPY ({str(sql)}) TO '{filename}' WITH (FORMAT xlsx, HEADER true"
                            if sheet:
                                stmt += f", SHEET '{sheet}'"
                            stmt += ")"
                            print(stmt)
                            con.execute(stmt)
                            # If band formatting requested, apply it
                            if band_by_group_col and filename.lower().endswith(".xlsx"):
                                _apply_band_formatting_xlsx(filename, sheet or "Sheet1",
                                                            band_by_group_col, band_wrap)
                            return None
                        except (
                            duckdb.BinderException,
                            duckdb.IOException,
                            duckdb.CatalogException,
                            duckdb.InternalException,
                        ) as ex:
                            # Fallback: materialize in Python and use pandas
                            df = con.execute(str(sql)).df()
                            try:
                                with pd.ExcelWriter(
                                    filename, engine="openpyxl"
                                ) as writer:
                                    df.to_excel(
                                        writer,
                                        sheet_name=sheet or "Sheet1",
                                        index=False,
                                    )
                                if band_by_group_col and filename.lower().endswith(".xlsx"):
                                    _apply_band_formatting_xlsx(filename, sheet or "Sheet1",
                                                                 band_by_group_col, band_wrap)
                            except (
                                PermissionError,
                                FileNotFoundError,
                                OSError,
                                ValueError,
                            ):
                                fallback = (
                                    filename
                                    if filename.lower().endswith(".csv")
                                    else (filename + ".csv")
                                )
                                df.to_csv(fallback, index=False)
                            return None

                    case InTransaction(subprog):
                        # DuckDB has transactional semantics; use BEGIN/COMMIT
                        try:
                            con.execute("BEGIN")
                            result = subprog._step(current)
                            con.execute("COMMIT")
                            return result
                        except Exception:
                            con.execute("ROLLBACK")
                            raise

                    case _:
                        return parent(intent, current)

            inner = Run(prog._step, perform)
            return inner._step(inner)
        finally:
            try:
                # Cleanly detach the SQLite-attached database if present to avoid
                # "database with name 'sqldb' already exists" on subsequent runs.
                con.execute("DETACH DATABASE sqldb;")
            except duckdb.CatalogException:
                # Ignore if not attached or already detached.
                pass
            finally:
                con.close()

    return Run(step, lambda i, c: c._perform(i, c))


def with_duckdb(subprog: Run[A]) -> Run[A]:
    """
    Context wrapper to run a sub-program in DuckDB mode.
    Assumes the environment has "duckdb_path" set.
    """
    return ask() >> (
        lambda env: run_duckdb(
            env["duckdb_path"], subprog, attach_sqlite_path=env["db_path"]
        )
    )
