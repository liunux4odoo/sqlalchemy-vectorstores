import pytest

import openai
from rich import print
import sqlalchemy as sa
from sqlalchemy_vectorstores import SqliteDatabase, SqliteVectorStore
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize


DB_URL = "sqlite:///:memory:"
# DB_URL = "sqlite:///test.db"
OPENAI_BASE_URL = "http://192.168.8.68:9997/v1"
OPENAI_API_KEY = "E"
EMBEDDING_MODEL = "bge-large-zh-v1.5"


client = openai.Client(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
def embed_func(text: str) -> list[float]:
    return client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    ).data[0].embedding

db = SqliteDatabase(DB_URL, fts_tokenizers={"jieba": JiebaTokenize()}, echo=False)
vs = SqliteVectorStore(db, dim=1024, embedding_func=embed_func, fts_tokenize="jieba")


query = "Alaqua Cox"
sentences1 = [
    "Capri-Sun is a brand of juice concentrate–based drinks manufactured by the German company Wild and regional licensees.",
    "George V was King of the United Kingdom and the British Dominions, and Emperor of India, from 6 May 1910 until his death in 1936.",
    "Alaqua Cox is a Native American (Menominee) actress.",
]
sentences2 = [
    "Shohei Ohtani is a Japanese professional baseball pitcher and designated hitter for the Los Angeles Dodgers of Major League Baseball.",
    "Tamarindo, also commonly known as agua de tamarindo, is a non-alcoholic beverage made of tamarind, sugar, and water.",
    "sqlalchemy-vectores 是一个通过 sqlalchemy 利用 sqlite 和 postgres 数据库实现向量检索和 BM25 全文检索功能的库。",
]


def test_version():
    with vs.connect() as conn:
        stmt = "select sqlite_version(), vec_version()"
        sqlite_version, vec_version = conn.execute(sa.text(stmt)).first()
        print(f"{sqlite_version=}")
        print(f"{vec_version=}")


def test_create():
    # add sources
    print("add sources")
    src_id1 = vs.add_source(url="file1.pdf", tags=["a", "b"], metadata={"path": "path1"})
    src_id2 = vs.add_source(url="file2.txt", tags=["c", "b"], metadata={"path": "path2"})

    # add documents
    print("add documents")
    for s in sentences1:
        vs.add_document(src_id=src_id1, content=s)

    for s in sentences2:
        vs.add_document(src_id=src_id2, content=s)

    # search sources by url
    print("search sources by url")
    r = vs.search_sources(vs.db.make_filter(vs.src_table.c.url, "file1.pdf"))
    print(r)
    assert isinstance(r, list) and len(r) == 1
    r = r[0]
    assert r["id"] == src_id1

    # search sources by metadata
    print("search sources by metadata")
    r = vs.search_sources(vs.db.make_filter(vs.src_table.c.metadata, "path2", "dict", "$.path"))
    print(r)
    assert isinstance(r, list) and len(r) == 1
    r = r[0]
    assert r["id"] == src_id2

    r = vs.search_sources(vs.db.make_filter(vs.src_table.c.metadata, "path%", "dict", "$.path"))
    print(r)
    assert isinstance(r, list) and len(r) == 2

    # search sources by tags
    print("search sources by tags")
    r = vs.get_sources_by_tags(tags_all=["b", "a"])
    print(r)
    assert len(r) == 1
    assert r[0]["tags"] == ["a", "b"]

    r = vs.get_sources_by_tags(tags_any=["b", "a"])
    print(r)
    assert len(r) == 2

    # upsert source with id
    print("upsert source with id")
    vs.upsert_source({"id": src_id1, "metadata": {"path": "path1", "added": True}})
    r = vs.get_source_by_id(src_id1)
    print(r)
    assert r["metadata"]["added"]

    # upsert source without id
    print("upsert source without id")
    src_id3 = vs.upsert_source({"url": "file3.docx", "metadata": {"path": "path3", "added": True}})
    r = vs.get_source_by_id(src_id3)
    print(r)
    assert r["metadata"]["path"] == "path3"
    assert r["url"] == "file3.docx"
    for s in sentences2:
        vs.add_document(src_id=src_id3, content=s)

    # list documents of source file
    print("list documents of source file")
    r = vs.get_documents_of_source(src_id3)
    print(r)
    assert len(r) == len(sentences2)

    # delete source
    print("delete source")
    r = vs.delete_source(src_id3)
    print(r)
    r = vs.get_source_by_id(src_id3)
    assert r is None
    r = vs.get_documents_of_source(src_id3)
    assert len(r) == 0

    # search by vector
    print("search by vector")
    r = vs.search_by_vector(query)
    print(r)
    assert len(r) == 3
    assert query in r[0]["content"]

    # search by vector with filters
    print("search by vector with filters")
    filters = [
        vs.db.make_filter(vs.src_table.c.url, "file1.pdf")
    ]
    r = vs.search_by_vector(query, filters=filters)
    print(r)
    assert query in r[0]["content"]

    # search by bm25
    print("search by bm25")
    r = vs.search_by_bm25(query)
    print(r)
    assert query in r[0]["content"]
