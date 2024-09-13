from __future__ import annotations

import abc
import uuid
import typing as t

import sqlalchemy as sa
from sqlalchemy_utils import ScalarListType


class VectorDatabase(abc.ABC):
    '''
    manage table creation and connection in database
    '''

    def __init__(
        self,
        db: str | sa.Engine,
        **db_kwds,
    ) -> None:
        super().__init__()
        if isinstance(db, sa.Engine):
            self.engine: sa.Engine = db
        else:
            self.engine: sa.Engine = sa.create_engine(db, **db_kwds)
        self.metadata = sa.MetaData()

    @property
    def tables(self) -> t.Dict[str, sa.Table]:
        '''
        all defined sqlalchemy Tables
        '''
        return self.metadata.tables

    def connect(self) -> sa.Connection:
        return self.engine.connect()

    def drop_table(self, table_name: str):
        with self.connect() as con:
            con.execute(sa.text(f"drop table if exists {table_name}"))
            con.commit()

    def create_src_table(self, table_name: str) -> sa.Table:
        '''
        table for document source
        '''
        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("id", sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True),
            sa.Column("url", sa.String),
            sa.Column("last_update_time", sa.DateTime, server_default=sa.func.now()),
            sa.Column("tags", ScalarListType(), default=[]),
            sa.Column("metadata", sa.JSON, default={}),
        )
        table.create(self.engine, checkfirst=True)
        return table

    def create_doc_table(self, table_name: str) -> sa.Table:
        '''
        table for document chunks
        '''
        table = sa.Table(
            table_name,
            self.metadata,
            sa.Column("id", sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True),
            sa.Column("src_id", sa.String(36)),
            sa.Column("content", sa.Text),
            sa.Column("type", sa.String(10)),
            sa.Column("target_id", sa.String(36)),
            sa.Column("metadata", sa.JSON, default={}),
        )
        table.create(self.engine, checkfirst=True)
        return table


    @abc.abstractmethod
    def create_fts_table(
        self,
        table_name: str,
        source_table: str,
        tokenize: str | None = None,
    ) -> sa.Table:
        '''
        table for full text search
        '''
        ...

    @abc.abstractmethod
    def create_vec_table(
        self,
        table_name: str,
        source_table: str,
        dim: int | None = None,
    ) -> sa.Table:
        '''
        table for vector search
        '''
        ...

    @abc.abstractmethod
    def make_filter(
        self,
        column: sa.Column,
        value: t.Any,
        type: t.Literal["id", "text", "list_any", "list_all", "dict"] = "text",
    ) -> sa.sql._typing.ColumnExpressionArgument:
        if type == "id":
            return (column == value)
        elif type == "text":
            return (column.ilike(value))
        elif type == "list_any": # TODO: ScalarListType will confuse if one element is part of another
            return sa.or_(*[column.contains(x) for x in value]) # add False cannot get correct results
        elif type == "list_all":
            return sa.and_(*[column.contains(x) for x in value]) # add True cannot get correct results
        else:
            raise RuntimeError(f"unsupported filter type: {type} for {column}")
