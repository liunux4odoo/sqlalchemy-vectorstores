from __future__ import annotations

import typing as t

import sqlalchemy as sa

from .base_async import AsyncBaseVectorStore


_PGV_STRATEGY = t.Literal[
    "l2_distance",
    "max_inner_product",
    "cosine_distance",
    "l1_distance",
    "hamming_distance",
    "jaccard_distance",
]


class AsyncPostgresVectorStore(AsyncBaseVectorStore):
    async def search_by_vector(
        self,
        query: str | t.List[float],
        top_k: int = 3,
        score_threshold: float | None = None,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
        strategy: _PGV_STRATEGY = "l2_distance",
    ) -> t.List[t.Dict]:
        if isinstance(query, str):
            assert self.embedding_func is not None
            query = await self.embedding_func(query)

        async with self.connect() as con:
            t1 = self.vec_table
            t2 = self.doc_table
            t3 = self.src_table
            stmt = (sa.select(t2, getattr(t1.c.embedding, strategy)(query).label("score"))
                    .join(t2, t1.c.doc_id==t2.c.id)
                    .join(t3, t2.c.src_id==t3.c.id)
                    .where(*filters)
                    .order_by("score")
                    .limit(top_k))
            docs = [x._asdict() for x in (await con.execute(stmt))]
        if score_threshold is not None:
            docs = [x for x in docs if x["score"] <= score_threshold]
        return docs

    async def search_by_bm25(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 2,
        filters: list[sa.sql._typing.ColumnExpressionArgument] = [],
    ) -> t.List[t.Dict]:
        async with self.connect() as con:
            t1 = self.fts_table
            t2 = self.doc_table
            t3 = self.src_table
            # make rank negative to compatible with sqlite fts
            rank = (-sa.func.ts_rank(t1.c.tsv, sa.func.to_tsquery(query))).label("score")
            stmt = (sa.select(t2, rank)
                    .join(t1, t1.c.id==t2.c.id)
                    .join(t3, t2.c.src_id==t3.c.id)
                    .where(*filters)
                    .order_by(rank)
                    .limit(top_k))
            docs = [x._asdict() for x in (await con.execute(stmt))]
        if score_threshold is not None:
            docs = [x for x in docs if x["score"] <= score_threshold]
        return docs

    async def add_document(
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
        doc_id = await super().add_document(
            src_id=src_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            type=type,
            target_id=target_id,
        )
        # add tsvector
        async with self.connect() as con:
            t = self.fts_table
            if callable(self.fts_tokenize):
                tsv = self.fts_tokenize(content)
            else:
                stmt = f"select to_tsvector('{self.fts_language}', '{content}');"
                tsv = (await con.execute(sa.text(stmt))).scalar()
            stmt = sa.insert(t).values(id=doc_id, tsv=tsv)
            await con.execute(stmt)
            await con.commit()
        return doc_id

    async def upsert_document(self, data: dict) -> str:
        doc_id = await super().upsert_document(data)
        if content := data.get("content"):
            async with self.connect() as con:
                t = self.fts_table
                stmt = (sa.update(t)
                        .where(t.c.id==doc_id)
                        .values(id=doc_id, content=content))
                await con.execute(stmt)
                await con.commit()
        return doc_id


    async def delete_document(self, id: str) -> t.Tuple[int, int]:
        '''
        delete a document chunk and it's vectors
        '''
        res = await super().delete_document(id)
        async with self.connect() as con:
            t = self.fts_table
            stmt = sa.delete(t).where(t.c.id==id)
            await con.execute(stmt)
            await con.commit()
        return res
