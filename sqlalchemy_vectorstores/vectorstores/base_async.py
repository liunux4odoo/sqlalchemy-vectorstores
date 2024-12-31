from __future__ import annotations

import abc
import asyncio
import typing as t

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from sqlalchemy_vectorstores.databases import AsyncBaseDatabase
from sqlalchemy_vectorstores.vectorstores.utils import _select_first_to_dict


class AsyncBaseVectorStore(abc.ABC):
    '''
    a simple vector store that support:
        - CRUD of documents
        - search documents by vector with filters
        - search documents by bm25 with filters
    '''
    def __init__(
        self,
        db: AsyncBaseDatabase,
        *,
        src_table: str = "",
        doc_table: str = "",
        fts_table: str = "",
        vec_table: str = "",
        words_table: str = "",
        table_prefix: str = "rag",
        fts_tokenize: t.Callable[[str], str] | None = None,
        fts_language: str = "english",
        embedding_func: t.Callable[[str], t.List[float]] | t.Callable[[t.List[str]], t.List[t.List[float]]] | None = None,
        dim: int | None = None,
        clear_existed: bool = False,
    ) -> None:
        self.db = db
        self._src_table = src_table or f"{table_prefix}_src"
        self._doc_table = doc_table or f"{table_prefix}_doc"
        self._fts_table = fts_table or f"{table_prefix}_fts"
        self._vec_table = vec_table or f"{table_prefix}_vec"
        self._words_table = words_table or f"{table_prefix}_words"
        self.fts_tokenize = fts_tokenize
        self.fts_language = fts_language
        self.embedding_func = embedding_func
        self.dim = dim
        self._con = None # TODO: optimize connection performance
        asyncio.run(self.init_database(clear_existed=clear_existed))

    async def init_database(self, clear_existed: bool = False):
        '''
        create all tables
        '''
        if self.embedding_func is not None and self.dim is None:
            self.dim = len(await self.embedding_func("hello world"))

        if clear_existed:
            await self.drop_all_tables()
        await self.db.create_src_table(self._src_table)
        await self.db.create_doc_table(self._doc_table)
        await self.db.create_fts_table(self._fts_table, self._doc_table, self.fts_tokenize)
        await self.db.create_vec_table(self._vec_table, self._doc_table, self.dim)
        await self.db.create_words_table(self._words_table)

    async def drop_all_tables(self):
        await self.db.drop_tables(
            self._src_table,
            self._doc_table,
            self._fts_table,
            self._vec_table,
            self._words_table
        )

    @property
    def src_table(self) -> sa.Table:
        return self.db.tables[self._src_table]

    @property
    def doc_table(self) -> sa.Table:
        return self.db.tables[self._doc_table]

    @property
    def fts_table(self) -> sa.Table:
        return self.db.tables[self._fts_table]

    @property
    def vec_table(self) -> sa.Table:
        return self.db.tables[self._vec_table]

    @property
    def words_table(self) -> sa.Table:
        return self.db.tables[self._words_table]

    def connect(self) -> AsyncConnection:
        return self.db.connect()

    async def add_source(
        self,
        url: str,
        *,
        tags: t.List[str] = [],
        metadata: dict = {},
    ) -> str:
        '''
        insert or update a document source to database
        '''
        data = {
            "url": url,
            "tags": tags,
            "metadata": metadata,
        }
        async with self.connect() as con:
            stmt = sa.insert(self.src_table).values(data)
            res = await con.execute(stmt)
            await con.commit()
            return res.inserted_primary_key[0]

    async def upsert_source(self, data: dict) -> str:
        async with self.connect() as con:
            t = self.src_table
            if id := data.pop("id", None):
                existed = await self.get_source_by_id(id)
                if existed is not None:
                    stmt = sa.update(t).values(data).where(t.c.id==id)
                    await con.execute(stmt)
                    await con.commit()
                    return id
        return await self.add_source(**data)

    async def clear_source(self, id: str) -> t.Tuple[int, int, int]:
        '''
        clear all documents and vectors of a source, keep the source record
        return the count of deleted documents/vectors
        '''
        doc_ids = [x["id"] for x in (await self.get_documents_of_source(id))]
        doc_count = await self.db.delete_by_ids(self.doc_table, doc_ids)
        vec_count = await self.db.delete_by_ids(self.vec_table, doc_ids, "doc_id")
        fts_count = await self.db.delete_by_ids(self.fts_table, doc_ids, "id")
        return doc_count, vec_count, fts_count

    async def delete_source(self, id: str) -> t.Tuple[int, int, int]:
        '''
        delete source and it's documents/vectors completely
        '''
        await self.db.delete_by_ids(self.src_table, id)
        return await self.clear_source(id)

    async def delete_source_by_url(self, url: str) -> t.Tuple[int, int, int]:
        filters = [
            self.db.make_filter(self.src_table.c.url, url)
        ]
        src_ids = [x["id"] for x in (await self.search_sources(filters))]
        for src_id in src_ids:
            await self.delete_source(src_id)

    async def search_sources(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[t.Dict]:
        async with self.connect() as con:
            stmt = self.src_table.select().where(*filters)
            return [x._asdict() for x in (await con.execute(stmt))]

    async def get_source_by_id(self, id: str) -> dict | None:
        async with self.connect() as con:
            t = self.src_table
            r = await con.execute(t.select().where(t.c.id==id))
            return _select_first_to_dict(r)

    async def get_sources_by_tags(
        self,
        *,
        tags_any: t.List[str] = [],
        tags_all: t.List[str] = [],
    ) -> t.List[t.Dict]:
        t = self.src_table
        expr1 = self.db.make_filter(t.c.tags, tags_any, "list_any")
        expr2 = self.db.make_filter(t.c.tags, tags_all, "list_all")
        return await self.search_sources(expr1, expr2)

    async def add_document(
        self,
        *,
        src_id: str,
        content: str,
        embedding: t.List[float] | None = None,
        metadata: dict = {},
        type: str | None = None,
        seq: int = -1,
        target_ids: list[str] = [],
    ) -> str:
        '''
        insert a document chunk to database, generate fts & vectors automatically
        '''
        data = {
            "src_id": src_id,
            "content": content,
            "metadata": metadata,
            "type": type,
            "seq": seq,
            "target_ids": target_ids,
        }
        if embedding is None and self.embedding_func is not None:
            embedding = await self.embedding_func(content)
        
        async with self.connect() as con:
            stmt = self.doc_table.insert().values(data)
            doc_id = (await con.execute(stmt)).inserted_primary_key[0]
            if embedding:
                stmt = self.vec_table.insert().values(doc_id=doc_id, embedding=embedding)
                await con.execute(stmt)
            await con.commit()
            return doc_id

    async def upsert_document(self, data: dict) -> str: # TODO: update vectors?
        async with self.connect() as con:
            t = self.doc_table
            if id := data.get("id", None):
                existed = await self.get_source_by_id(id)
                if existed is not None:
                    stmt = sa.update(t).values(data).where(t.c.id==id)
                    await con.execute(stmt)
                    await con.commit()
                    return id
        return await self.add_document(**data)

    async def delete_documents(self, ids: t.List[str]) -> t.Tuple[int, int, int]:
        '''
        delete a document chunk and it's vectors
        '''
        doc_count = await self.db.delete_by_ids(self.doc_table, ids, "id")
        vec_count = await self.db.delete_by_ids(self.vec_table, id, "doc_id")
        fts_count = await self.db.delete_by_ids(self.fts_table, id, "id")
        return doc_count, vec_count, fts_count

    async def search_documents(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[t.Dict]:
        if len(filters) == 1 and isinstance(filters[0], list):
            filters = filters[0]
        async with self.connect() as con:
            stmt = self.doc_table.select().where(*filters)
            return [x._asdict() for x in (await con.execute(stmt))]

    async def get_document_by_ids(self, ids: t.List[str]) -> t.List[dict]:
        async with self.connect() as con:
            t = self.doc_table
            r = await con.execute(t.select().where(t.c.id==id))
            return [x._asdict() for x in r]

    async def get_documents_of_source(self, source_id: str) -> t.List[t.Dict]:
        expr = self.db.make_filter(self.doc_table.c.src_id, source_id, "id")
        return await self.search_documents(expr)

    async def get_documents_by_meta(self, kw: t.Dict) -> t.List[t.Dict]:
        expr = [self.db.make_filter(self.doc_table.c.metadata, v, "dict", k) for k,v in kw.items()]
        return await self.search_documents(expr)

    @abc.abstractmethod
    async def search_by_vector(
        self,
        query: str | t.List[float],
        top_k: int = 3,
        score_threshold: float | None = None,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        ...

    @abc.abstractmethod
    async def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 2,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        ...

    async def get_stop_words(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[str]:
        async with self.connect() as con:
            t = self.words_table
            stmt = sa.select(t.c.word).where(t.c.status==0).where(*filters)
            return [r[0] for r in (await con.execute(stmt))]

    async def get_user_dict(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[dict]:
        async with self.connect() as con:
            t = self.words_table
            stmt = sa.select(t.c.word, t.c.freq, t.c.tag).where(t.c.status==1).where(*filters)
            return [r._asdict() for r in (await con.execute(stmt))]
