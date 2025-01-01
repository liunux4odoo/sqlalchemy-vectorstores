import asyncio
import pytest

import openai
from rich import print
from sqlalchemy_vectorstores import AsyncPostgresDatabase, AsyncPostgresVectorStore


DB_URL = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres"
OPENAI_BASE_URL = "http://192.168.8.68:9997/v1"
OPENAI_API_KEY = "E"
EMBEDDING_MODEL = "bge-large-zh-v1.5"


client = openai.AsyncClient(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
async def embed_func(text: str) -> list[float]:
    return (await client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )).data[0].embedding

db = AsyncPostgresDatabase(DB_URL, echo=False)
vs = AsyncPostgresVectorStore(db, dim=1024, embedding_func=embed_func, clear_existed=True)

query = "Alaqua Cox"
sentences1 = [
    "Capri-Sun is a brand of juice concentrate–based drinks manufactured by the German company Wild and regional licensees.",
    "George V was King of the United Kingdom and the British Dominions, and Emperor of India, from 6 May 1910 until his death in 1936.",
    "Alaqua Cox is a Native American (Menominee) actress.",
]
sentences2 = [
    "Shohei Ohtani is a Japanese professional baseball pitcher and designated hitter for the Los Angeles Dodgers of Major League Baseball.",
    "Tamarindo, also commonly known as agua de tamarindo, is a non-alcoholic beverage made of tamarind, sugar, and water.",
]


@pytest.mark.asyncio
async def test_create():
    # add sources
    print("add sources")
    src_id1 = await vs.add_source(src="file1.pdf", tags=["a", "b"], metadata={"path": "path1"})
    src_id2 = await vs.add_source(src="file2.txt", tags=["c", "b"], metadata={"path": "path2"})

    # add documents
    print("add documents")
    for s in sentences1:
        await vs.add_document(src_id=src_id1, content=s)

    for s in sentences2:
        await vs.add_document(src_id=src_id2, content=s)

    # search sources by url
    print("search sources by url")
    r = await vs.search_sources(vs.db.make_filter(vs.src_table.c.src, "file1.pdf"))
    print(r)
    assert isinstance(r, list) and len(r) == 1
    r = r[0]
    assert r["id"] == src_id1

    # search sources by metadata
    print("search sources by metadata")
    r = await vs.search_sources(vs.db.make_filter(vs.src_table.c.metadata, "path2", "dict", "path"))
    print(r)
    assert isinstance(r, list) and len(r) == 1
    r = r[0]
    assert r["id"] == src_id2

    r = await vs.search_sources(vs.db.make_filter(vs.src_table.c.metadata, "path%", "dict", "path"))
    print(r)
    assert isinstance(r, list) and len(r) == 2

    # search sources by tags
    print("search sources by tags")
    r = await vs.get_sources_by_tags(tags_all=["b", "a"])
    print(r)
    assert len(r) == 1
    assert r[0]["tags"] == ["a", "b"]

    r = await vs.get_sources_by_tags(tags_any=["b", "a"])
    print(r)
    assert len(r) == 2

    # upsert source with id
    print("upsert source with id")
    await vs.upsert_source({"id": src_id1, "metadata": {"path": "path1", "added": True}})
    r = await vs.get_source_by_id(src_id1)
    print(r)
    assert r["metadata"]["added"]

    # upsert source without id
    print("upsert source without id")
    src_id3 = await vs.upsert_source({"src": "file3.docx", "metadata": {"path": "path3", "added": True}})
    r = await vs.get_source_by_id(src_id3)
    print(r)
    assert r["metadata"]["path"] == "path3"
    assert r["src"] == "file3.docx"
    for s in sentences2:
        await vs.add_document(src_id=src_id3, content=s)

    # list documents of source file
    print("list documents of source file")
    r = await vs.get_documents_of_source(src_id3)
    print(r)
    assert len(r) == len(sentences2)

    # delete source
    print("delete source")
    r = await vs.delete_source(src_id3)
    print(r)
    r = await vs.get_source_by_id(src_id3)
    assert r is None
    r = await vs.get_documents_of_source(src_id3)
    assert len(r) == 0

    # search by vector
    print("search by vector")
    r = await vs.search_by_vector(query)
    print(r)
    assert len(r) == 3
    assert query in r[0]["content"]

    # search by vector with filters
    print("search by vector with filters")
    filters = [
        vs.db.make_filter(vs.src_table.c.src, "file1.pdf")
    ]
    r = await vs.search_by_vector(query, filters=filters)
    print(r)
    assert query in r[0]["content"]

    # search by bm25
    print("search by bm25")
    r = await vs.search_by_bm25(query.replace(" ", " & "))
    print(r)
    assert query in r[0]["content"]
