"""
Intent and eliminator for SQL (database) effects.
"""
#pylint:disable=W0212
from dataclasses import dataclass
from typing import TypeVar
from typing import Any

import sqlite3

from .array import Array
from .run import Run, _unhandled
from .string import String

class SQL(String):
    """
    Represents a SQL string.
    """

type SQLParam = String | int
class SQLParams(Array[SQLParam]):
    """
    Represents a list of SQL parameters.
    """

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
    return Run(lambda self: self._perform(SqlQuery(sql, params), self),
               _unhandled)

def sql_exec(sql: SQL, params: SQLParams = SQLParams(())) -> Run[None]:
    """
    Smart constructor for SQL execution with intent.
    """
    return Run(lambda self: self._perform(SqlExec(sql, params), self),
               _unhandled)

def sql_script(sql: SQL) -> "Run[None]":
    """
    Smart constructor for SQL scripts with intent.
    """
    return Run(lambda self: self._perform(SqlScript(sql), self), _unhandled)

def in_transaction(prog: "Run[A]") -> "Run[A]":
    """
    Smart constructor for database transactions with intent.
    """
    return Run(lambda self: self._perform(InTransaction(prog), self),
               _unhandled)

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
                        cur = con.execute(sql, params)
                        rows = cur.fetchall()
                        cur.close()
                        return rows

                    case SqlExec(sql, params):
                        con.execute(sql, params)
                        return None

                    case SqlScript(sql):
                        con.executescript(sql)
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
