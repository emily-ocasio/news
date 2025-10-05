"""
Intent and eliminator for SQL (database) effects.
"""

# pylint:disable=W0212
from dataclasses import dataclass
from typing import TypeVar
from typing import Any

import sqlite3
import duckdb

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


A = TypeVar("A")


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
