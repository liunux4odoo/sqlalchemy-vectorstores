import openai
from sqlalchemy_vectorstores import SqliteDatabase, SqliteVectorStore
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize


DB_URL = "sqlite:///:memory:"
OPENAI_BASE_URL = "http://192.168.8.68:9997/v1" # local xinference server
OPENAI_API_KEY = "E"
EMBEDDING_MODEL = "bge-large-zh-v1.5"


client = openai.Client(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
def embed_func(text: str) -> list[float]:
    return client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    ).data[0].embedding

# Using sync sqlite database. you can use other 3 combinations.
db = SqliteDatabase(DB_URL, fts_tokenizers={"jieba": JiebaTokenize()}, echo=False)
vs = SqliteVectorStore(db, dim=1024, embedding_func=embed_func, fts_tokenize="jieba")


query = "Alaqua Cox"
sentences = [
    "Capri-Sun is a brand of juice concentrate–based drinks manufactured by the German company Wild and regional licensees.",
    "George V was King of the United Kingdom and the British Dominions, and Emperor of India, from 6 May 1910 until his death in 1936.",
    "Alaqua Cox is a Native American (Menominee) actress.",
    "Shohei Ohtani is a Japanese professional baseball pitcher and designated hitter for the Los Angeles Dodgers of Major League Baseball.",
    "Tamarindo, also commonly known as agua de tamarindo, is a non-alcoholic beverage made of tamarind, sugar, and water.",
]


# add sources
src_id = vs.add_source(url="file1.pdf", tags=["a", "b"], metadata={"path": "path1"})

# add documents
for s in sentences:
    vs.add_document(src_id=src_id, content=s)

# search sources by url
r = vs.search_sources(vs.db.make_filter(vs.src_table.c.url, "file1.pdf"))
print(r)

# search sources by metadata
# vs.db_make_filter is a helper method to build sqlalchemy expressions.
r = vs.search_sources(vs.db.make_filter(vs.src_table.c.metadata, "path2", "dict", "$.path"))
print(r)

# search sources by tags
r = vs.get_sources_by_tags(tags_all=["b", "a"])
print(r)

r = vs.get_sources_by_tags(tags_any=["b", "a"])
print(r)

# upsert source with id - update
vs.upsert_source({"id": src_id, "metadata": {"path": "path1", "added": True}})
r = vs.get_source_by_id(src_id)
print(r)

# upsert source without id - insert
src_id3 = vs.upsert_source({"url": "file3.docx", "metadata": {"path": "path3", "added": True}})
r = vs.get_source_by_id(src_id3)
print(r)

# list documents of source file
print("list documents of source file")
r = vs.get_documents_of_source(src_id3)

# delete source and documents/tsvector/embeddings belongs to it.
r = vs.delete_source(src_id3)

# search by vector
r = vs.search_by_vector(query)
print(r)

# search by vector with filters
filters = [
    vs.db.make_filter(vs.src_table.c.url, "file1.pdf")
]
r = vs.search_by_vector(query, filters=filters)
print(r)

# search by bm25
r = vs.search_by_bm25(query)
print(r)
