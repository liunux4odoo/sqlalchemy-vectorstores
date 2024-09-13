from __future__ import annotations

import abc
import typing as t

import sqlalchemy as sa

from sqlalchemy_vectorstores.databases import VectorDatabase
from sqlalchemy_vectorstores.vectorstores.utils import _select_first_to_dict


class BaseVectorStore(abc.ABC):
    '''
    a simple vector store that support:
        - CRUD of document tables
        - search documents by vector with filters
        - search documents by bm25 with filters
    '''
    def __init__(
        self,
        db: VectorDatabase,
        *,
        src_table: str = "rag_src",
        doc_table: str = "rag_doc",
        fts_table: str = "rag_fts",
        vec_table: str = "rag_vec",
        fts_tokenize: str = "porter",
        embedding_func: t.Callable[[str], t.List[float]] | None = None,
        dim: int | None = None,
    ) -> None:
        self.db = db
        self._src_table = src_table
        self._doc_table = doc_table
        self._fts_table = fts_table
        self._vec_table = vec_table
        self.fts_tokenize = fts_tokenize
        self.embedding_func = embedding_func
        self.dim = dim

        self.init_database()

    def init_database(self):
        '''
        create all tables
        '''
        if self.embedding_func is not None and self.dim is None:
            self.dim = len(self.embedding_func("hello world"))
            
        self.db.create_src_table(self._src_table)
        self.db.create_doc_table(self._doc_table)
        self.db.create_fts_table(self._fts_table, self._doc_table, self.fts_tokenize)
        self.db.create_vec_table(self._vec_table, self._doc_table, self.dim)

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

    def connect(self) -> sa.Connection:
        return self.db.connect()

    def add_source(
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
        with self.connect() as con:
            stmt = sa.insert(self.src_table).values(data)
            res = con.execute(stmt)
            con.commit()
            return res.inserted_primary_key[0]

    def upsert_source(self, data: dict) -> str:
        with self.connect() as con:
            t = self.src_table
            if id := data.pop("id", None):
                existed = self.get_source_by_id(id)
                if existed is not None:
                    stmt = sa.update(t).values(data).where(t.c.id==id)
                    con.execute(stmt)
                    con.commit()
                    return id
        return self.add_source(**data)

    def clear_source(self, id: str) -> t.Tuple[int, int]:
        '''
        clear all documents and vectors of a source, keep the source record
        return the count of deleted documents/vectors
        '''
        with self.connect() as con:
            # delete documents
            t = self.doc_table
            doc_ids = [x["id"] for x in self.get_documents_of_source(id)]
            stmt = sa.delete(t).where(t.c.src_id==id)
            res1 = con.execute(stmt)

            # delete vectors
            t = self.vec_table
            stmt = sa.delete(t).where(t.c.doc_id.in_(doc_ids))
            res2 = con.execute(stmt)

            con.commit()
            return res1.rowcount, res2.rowcount

    def delete_source(self, id: str) -> t.Tuple[int, int, int]:
        '''
        delete source and it's documents/vectors completely
        '''
        with self.connect() as con:
            # delete source
            t = self.src_table
            stmt = sa.delete(t).where(t.c.id==id)
            res1 = con.execute(stmt)
            con.commit()
        return (res1.rowcount,) + self.clear_source(id)

    def search_sources(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[t.Dict]:
        with self.connect() as con:
            stmt = self.src_table.select().where(*filters)
            return [x._asdict() for x in con.execute(stmt)]

    def get_source_by_id(self, id: str) -> dict | None:
        with self.connect() as con:
            t = self.src_table
            r = con.execute(t.select().where(t.c.id==id))
            return _select_first_to_dict(r)

    def get_sources_by_tags(
        self,
        *,
        tags_any: t.List[str] = [],
        tags_all: t.List[str] = [],
    ) -> t.List[t.Dict]:
        t = self.src_table
        expr1 = self.db.make_filter(t.c.tags, tags_any, "list_any")
        expr2 = self.db.make_filter(t.c.tags, tags_all, "list_all")
        return self.search_sources(expr1, expr2)

    def add_document(
        self,
        *,
        src_id: str,
        content: str,
        embedding: t.List[float] | None = None,
        metadata: dict = {},
        type: str | None = None,
        target_id: str | None = None,
    ) -> str:
        '''
        insert a document chunk to database, generate fts & vectors automatically
        '''
        data = {
            "src_id": src_id,
            "content": content,
            "metadata": metadata,
            "type": type,
            "target_id": target_id,
        }
        if embedding is None and self.embedding_func is not None:
            embedding = self.embedding_func(content)
        
        with self.connect() as con:
            stmt = self.doc_table.insert().values(data)
            doc_id = con.execute(stmt).inserted_primary_key[0]
            stmt = self.vec_table.insert().values(doc_id=doc_id, embedding=embedding)
            con.execute(stmt)
            con.commit()
            return doc_id

    def upsert_document(self, data: dict) -> str: # TODO: update vectors?
        with self.connect() as con:
            t = self.doc_table
            if id := data.get("id", None):
                existed = self.get_source_by_id(id)
                if existed is not None:
                    stmt = sa.update(t).values(data).where(t.c.id==id)
                    con.execute(stmt)
                    con.commit()
                    return id
        return self.add_document(**data)

    def delete_document(self, id: str) -> t.Tuple[int, int]:
        '''
        delete a document chunk and it's vectors
        '''
        with self.connect() as con:
            # delete 
            t = self.doc_table
            stmt = sa.delete(t).where(t.c.id==id)
            res1 = con.execute(stmt)

            # delete vectors
            t = self.vec_table
            stmt = sa.delete(t).where(t.c.doc_id==id)
            res2 = con.execute(stmt)

            con.commit()
            return res1.rowcount, res2.rowcount

    def search_documents(self, *filters: sa.sql._typing.ColumnExpressionArgument) -> t.List[t.Dict]:
        with self.connect() as con:
            stmt = self.doc_table.select().where(*filters)
            return [x._asdict() for x in con.execute(stmt)]

    def get_document_by_id(self, id: str) -> dict:
        with self.connect() as con:
            t = self.doc_table
            r = con.execute(t.select().where(t.c.id==id))
            return _select_first_to_dict(r)

    def get_documents_of_source(self, source_id: str) -> t.List[t.Dict]:
        expr = self.db.make_filter(self.doc_table.c.src_id, source_id, "id")
        return self.search_documents(expr)

    @abc.abstractmethod
    def search_by_vector(
        self,
        query: str | t.List[float],
        top_k: int = 3,
        score_threshold: float | None = None,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        ...

    @abc.abstractmethod
    def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 2,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        ...
