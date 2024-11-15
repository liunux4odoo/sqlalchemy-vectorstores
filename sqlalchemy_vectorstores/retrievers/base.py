from __future__ import annotations

import abc
import typing as t

from sqlalchemy_vectorstores import BaseVectorStore, DocType
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize


class BaseRetriever(abc.ABC):
    def __init__(self, vs: BaseVectorStore) -> None:
        self.vs = vs
        self.tokenize = JiebaTokenize()

    def count_words(self, text: str) -> int:
        return len(self.tokenize.cut_words(text))

    def retrieve(
        self,
        query: str,
        query_bm25: str | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
        max_words: int | None = None,
        min_words: int | None = None,
        score_threshold_vector: float | None = None,
        score_threshold_bm25: float | None = -0.1,
        src_id: str | None = None,
        src_url: str | None = None,
        src_metadata: t.Dict | None = None,
        src_tags_all: t.List[str] | None = None,
        src_tags_any: t.List[str] | None = None,
        doc_metadata: t.Dict | None = None,
        doc_types: t.List[DocType] | None = None,
    ) -> t.List[t.Dict]:
        """
        retrieve documents from vectorstore

        Args:
            query (str): user query string
            query_bm25 (str | None, optional): custom bm25 query string instead of using default words cut. Defaults to None.
            top_k (int | None, optional): _description_. Defaults to None.
            fetch_k (int | None, optional): _description_. Defaults to None.
            max_words (int | None, optional): _description_. Defaults to None.
            min_words (int | None, optional): _description_. Defaults to None.
            score_threshold_vector (float | None, optional): _description_. Defaults to None.
            score_threshold_bm25 (float | None, optional): _description_. Defaults to -0.1.
            src_id (str | None, optional): _description_. Defaults to None.
            src_url (str | None, optional): _description_. Defaults to None.
            src_metadata (t.Dict | None, optional): _description_. Defaults to None.
            src_tags_all (t.List[str] | None, optional): _description_. Defaults to None.
            src_tags_any (t.List[str] | None, optional): _description_. Defaults to None.
            doc_metadata (t.Dict | None, optional): _description_. Defaults to None.
            doc_types (t.List[str] | None, optional): _description_. Defaults to None.

        Returns:
            t.List[t.Dict]: a list of documents
        """