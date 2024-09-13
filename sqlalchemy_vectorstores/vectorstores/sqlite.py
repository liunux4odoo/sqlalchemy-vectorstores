from __future__ import annotations

import typing as t

import sqlalchemy as sa

from .base import BaseVectorStore


class SqliteVectorStore(BaseVectorStore):
    def search_by_vector(
        self,
        query: str | t.List[float],
        top_k: int = 3,
        score_threshold: float | None = None,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        if isinstance(query, str):
            assert self.embedding_func is not None
            query = self.embedding_func(query)

        with self.connect() as con:
            t1 = self.vec_table
            t2 = self.doc_table
            t3 = self.src_table
            stmt = (sa.select(t2, t1.c.distance.label("score"))
                    .join(t2, t1.c.doc_id==t2.c.id)
                    .join(t3, t2.c.src_id==t3.c.id)
                    .where(*filters)
                    .where(t1.c.embedding.match(query), sa.text(f"k={top_k}")))
            docs = [x._asdict() for x in con.execute(stmt)]
        if score_threshold is not None:
            docs = [x for x in docs if x["score"] <= score_threshold]
        return docs

    def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 2,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        with self.connect() as con:
            t1 = self.fts_table
            t2 = self.doc_table
            t3 = self.src_table
            stmt = (sa.select(t2, t1.c.rank.label("score"))
                    .join(t2, t1.c.id==t2.c.id)
                    .join(t3, t2.c.src_id==t3.c.id)
                    .where(*filters)
                    .where(t1.c.content.match(query))
                    .limit(top_k))
            docs = [x._asdict() for x in con.execute(stmt)]
        if score_threshold is not None:
            docs = [x for x in docs if x["score"] <= score_threshold]
        return docs
