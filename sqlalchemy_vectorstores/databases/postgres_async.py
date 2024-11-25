from __future__ import annotations

import asyncio
import sys
import uuid
import typing as t

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy_utils import ScalarListType

from .base_async import AsyncBaseDatabase


# latest psycopg fails on windows in default async loop
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class AsyncPostgresDatabase(AsyncBaseDatabase):
    '''
    use the postgres database to store documents, embeddings and tsvector
    '''
    def __init__(
        self,
        db: str | AsyncEngine,
        **db_kwds,
    ) -> None:
        super().__init__(db, **db_kwds)
        asyncio.run(self._init_database())

    async def _init_database(self):
        async with self.engine.connect() as con:
            await con.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await con.commit()
        # enable psycopg2 async not work
        # @sa_event.listens_for(self.engine.sync_engine, "do_connect")
        # def receive_connect(dialect, con_rec, cargs, cparams):
        #     cparams["async"] = 1

    
    async def create_src_table(self, table_name: str) -> sa.Table:
        '''
        table for document source
        '''
        from sqlalchemy.dialects.postgresql import JSONB

        if table_name in self.tables:
            return self.tables[table_name]

        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("id", sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True),
            sa.Column("url", sa.String),
            sa.Column("last_update_time", sa.DateTime, server_default=sa.func.now()),
            sa.Column("tags", ScalarListType(), default=[]),
            sa.Column("metadata", JSONB, default={}),
        )
        async with self.connect() as con:
            await con.run_sync(table.create, checkfirst=True)
        return table

    async def create_doc_table(self, table_name: str) -> sa.Table:
        '''
        table for document chunks
        '''
        from sqlalchemy.dialects.postgresql import JSONB

        if table_name in self.tables:
            return self.tables[table_name]

        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("id", sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True),
            sa.Column("src_id", sa.String(36)),
            sa.Column("content", sa.Text),
            sa.Column("type", sa.String(10)),
            sa.Column("target_id", sa.String(36)),
            sa.Column("metadata", JSONB, default={}),
        )
        async with self.connect() as con:
            await con.run_sync(table.create, checkfirst=True)
        return table

    async def create_fts_table(
        self,
        table_name: str,
        source_table: str,
        tokenize: str | None = None,
    ) -> sa.Table:
        '''
        table for full text search in postgres.
        '''
        from sqlalchemy.dialects.postgresql import TSVECTOR

        if table_name in self.tables:
            return self.tables[table_name]

        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("id", sa.String(36)),
            sa.Column("tsv", TSVECTOR),
            sa.Index(f"idx_{table_name}_tsv", "tsv", postgresql_using="gin"),
        )
        async with self.connect() as con:
            await con.run_sync(table.create, checkfirst=True)
        return table

    async def create_vec_table(
        self,
        table_name: str,
        source_table: str,
        dim: int | None = None,
    ) -> sa.Table:

        '''
        table for vector search in postgres using pgvector
        '''
        from pgvector.sqlalchemy import Vector

        if table_name in self.tables:
            return self.tables[table_name]

        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("doc_id", sa.String(36)),
            sa.Column("embedding", Vector(dim)),
        )
        async with self.connect() as con:
            await con.run_sync(table.create, checkfirst=True)
        return table

    def make_filter(
        self,
        column: sa.Column,
        value: t.Any,
        type: t.Literal["id", "text", "list_any", "list_all", "dict"] = "text",
        json_key: str = "",
    ) -> sa.sql._typing.ColumnExpressionArgument:
        """
        helper method to build sqlalchemy filter expressions.

        Args:
            column (sa.Column): the Column used to filter
            value (t.Any): filter by value
            type : filter type. Defaults to "text".
                    id: sa.Column==value
                    text: sa.Column.ilike(value)
                    list_any: sa.or_(sa.Column.contains(x) for x in value)
                    list_all: sa.and_(sa.Column.contains(x) for x in value)
                    dict: for metadata, sa.Column[json_key] [==, ilike] value
        Returns:
            sa.sql._typing.ColumnExpressionArgument
        """
        if type == "dict":
            if isinstance(value, str):
                return (column[json_key].as_string().ilike(value))
            else:
                return (column[json_key] == value)
        else:
            return super().make_filter(column, value, type)
