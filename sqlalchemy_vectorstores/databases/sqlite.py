from __future__ import annotations

import textwrap
import typing as t

import sqlalchemy as sa
from sqlalchemy import event as sa_event

from .base import VectorDatabase
from .sa_types import SqliteVector

if t.TYPE_CHECKING:
    import sqlite3
    from sqlitefts import fts5


class SqliteDatabase(VectorDatabase):
    '''
    use the sqlite database with some customizations:
        - custom sql functions
        - custom fts tokenizers
    '''

    def __init__(
        self,
        db: str | sa.Engine,
        *,
        fts_tokenizers: t.Dict[str, fts5.FTS5Tokenizer] = {},
        custom_functions: t.Dict[str, t.Callable] = {},
        **db_kwds,
    ) -> None:
        super().__init__(db, **db_kwds)

        # create an async sqlite engine with customizations
        import inspect
        import sqlite_vec
        from sqlitefts import fts5

        @sa_event.listens_for(self.engine, "connect")
        def on_connect(con: sqlite3.Connection, con_rec):
            # enable custom fts5 tokenizer
            for name, tokenizer in fts_tokenizers.items():
                fts5.register_tokenizer(con, name, tokenizer)

            # register custom functions
            for name, func in custom_functions.items():
                con.create_function(name, len(inspect.signature(func)), func)

            # load sqlite-vec extension
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            con.enable_load_extension(False)

    def create_fts_table(self, table_name: str, source_table: str, tokenize: str = "porter") -> sa.Table:
        '''
        table for full text search in sqlite
        '''
        columns = ["id", "content"]
        with self.engine.connect() as con:
            create_fts_sql = (
                textwrap.dedent(
                    """
                CREATE VIRTUAL TABLE IF NOT EXISTS [{fts_table_name}]
                    USING FTS5 (
                    {columns},{tokenize}
                    content=[{table}]
                )
            """
                )
                .strip()
                .format(
                    table=source_table,
                    fts_table_name=table_name,
                    columns=", ".join("[{}]".format(c) for c in columns),
                    tokenize="\n    tokenize='{}',".format(tokenize) if tokenize else "",
                )
            )
            con.execute(sa.text(create_fts_sql))

            # create triggers
            old_cols = ", ".join("old.[{}]".format(c) for c in columns)
            new_cols = ", ".join("new.[{}]".format(c) for c in columns)
            triggers = (
                textwrap.dedent(
                    """
                CREATE TRIGGER IF NOT EXISTS [{fts_table_name}_ai] AFTER INSERT ON [{table}] BEGIN
                  INSERT INTO [{fts_table_name}] (rowid, {columns}) VALUES (new.rowid, {new_cols});
                END;
                CREATE TRIGGER IF NOT EXISTS [{fts_table_name}_ad] AFTER DELETE ON [{table}] BEGIN
                  INSERT INTO [{fts_table_name}] ([{fts_table_name}], rowid, {columns}) VALUES('delete', old.rowid, {old_cols});
                END;
                CREATE TRIGGER IF NOT EXISTS [{fts_table_name}_au] AFTER UPDATE ON [{table}] BEGIN
                  INSERT INTO [{fts_table_name}] ([{fts_table_name}], rowid, {columns}) VALUES('delete', old.rowid, {old_cols});
                  INSERT INTO [{fts_table_name}] (rowid, {columns}) VALUES (new.rowid, {new_cols});
                END;
            """
                )
                .strip()
                .format(
                    table=source_table,
                    fts_table_name=table_name,
                    columns=", ".join("[{}]".format(c) for c in columns),
                    old_cols=old_cols,
                    new_cols=new_cols,
                )
            )
            # con._dbapi_connection.executescript(triggers)
            for trigger in [x for x in triggers.split("END;") if x.strip()]:
                con.execute(sa.text(trigger + "END;"))

            # self.metadata.reflect(self.engine, only=[table_name])
            table = sa.Table(
                table_name,
                self.metadata,
                sa.Column("id", sa.String(36)),
                sa.Column("content", sa.Text),
                sa.Column("rank", sa.Float),
            )
            return table

    def create_vec_table(self, table_name: str, source_table: str, dim: int) -> sa.Table:
        '''
        table for vector search in sqlite using sqlite-vec
        '''
        columns = ["embedding"]
        with self.engine.connect() as con:
            create_vec_sql = (textwrap.dedent(
                """
                    CREATE VIRTUAL TABLE IF NOT EXISTS [{vec_table_name}]
                    USING vec0(
                    doc_id TEXT PRIMARY KEY,
                    embedding FLOAT[{dim}]
                    );
                """
            )
            .strip()
            .format(
                vec_table_name=table_name,
                dim=dim,
            ))
            con.execute(sa.text(create_vec_sql))

            # # create triggers
            # old_cols = ", ".join("old.[{}]".format(c) for c in columns)
            # new_cols = ", ".join("new.[{}]".format(c) for c in columns)
            # triggers = (
            #     textwrap.dedent(
            #         """
            #     CREATE TRIGGER IF NOT EXISTS [{table}_ai] AFTER INSERT ON [{table}] BEGIN
            #       INSERT INTO [{vec_table_name}] (rowid, {columns}) VALUES (new.rowid, {new_cols});
            #     END;
            #     CREATE TRIGGER IF NOT EXISTS [{table}_ad] AFTER DELETE ON [{table}] BEGIN
            #       INSERT INTO [{vec_table_name}] ([{vec_table_name}], rowid, {columns}) VALUES('delete', old.rowid, {old_cols});
            #     END;
            #     CREATE TRIGGER IF NOT EXISTS [{table}_au] AFTER UPDATE ON [{table}] BEGIN
            #       INSERT INTO [{vec_table_name}] ([{vec_table_name}], rowid, {columns}) VALUES('delete', old.rowid, {old_cols});
            #       INSERT INTO [{vec_table_name}] (rowid, {columns}) VALUES (new.rowid, {new_cols});
            #     END;
            # """
            #     )
            #     .strip()
            #     .format(
            #         table=table_name,
            #         vec_table_name=table_name,
            #         columns=", ".join("[{}]".format(c) for c in columns),
            #         old_cols=old_cols,
            #         new_cols=new_cols,
            #     )
            # )
            # con._dbapi_connection.executescript(triggers)
            # for trigger in [x for x in triggers.split("END;") if x.strip()]:
            #     con.execute(sa.text(trigger + "END;"))

            # self.metadata.reflect(self.engine, only=[table_name])
            table = sa.Table(table_name, self.metadata,
                             sa.Column("doc_id", sa.String(36)),
                             sa.Column("embedding", SqliteVector(dim)),
                             sa.Column("distance", sa.Float))
            return table

    def make_filter(
        self,
        column: sa.Column,
        value: t.Any,
        type: t.Literal["id", "text", "list_any", "list_all", "dict"] = "text",
        json_path: str = "",
    ) -> sa.sql._typing.ColumnExpressionArgument:
        if type == "dict":
            if isinstance(value, str):
                return (sa.func.json_extract(column, json_path).ilike(value))
            else:
                return (sa.func.json_extract(column, json_path) == value)
        else:
            return super().make_filter(column, value, type)
