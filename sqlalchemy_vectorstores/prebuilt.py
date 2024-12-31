import typing as T
from functools import lru_cache

from sqlalchemy_vectorstores import (SqliteDatabase, SqliteVectorStore,
                                     AsyncSqliteDatabase, AsyncSqliteVectorStore,
                                     PostgresDatabase, PostgresVectorStore,
                                     AsyncPostgresDatabase, AsyncPostgresVectorStore)
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize


@T.overload
def create_embedding_func(
    model: str = "bge-large-zh-v1.5",
    base_url: str = "http://127.0.0.1:9997/v1",
    api_key: str = "EMPTY",
    batch_size: int = 10,
    is_async: bool = False,
) -> T.Callable:
    ...


@lru_cache()
def create_embedding_func(
    model: str = "bge-large-zh-v1.5",
    base_url: str = "http://127.0.0.1:9997/v1",
    api_key: str = "EMPTY",
    batch_size: int = 10,
    is_async: bool = False,
) -> T.Callable:
    import openai

    if batch_size <= 0:
        batch_size = 1000000

    params = dict(
        base_url=base_url,
        api_key=api_key,
    )
    client = openai.Client(**params)
    async_client = openai.AsyncClient(**params)

    @T.overload
    def embed_func(text: str) -> list[float]: ...

    @T.overload
    def embed_func(text: list[str]) -> list[list[float]]: ...

    def embed_func(text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            resp = client.embeddings.create(input=text, model=model)
            return resp.data[0].embedding
        else:
            res = []
            for i in range(0, len(text), batch_size):
                resp = client.embeddings.create(input=text[i:(i+1)*batch_size], model=model)
                res += [x.embedding for x in resp.data]
            return res

    @T.overload
    async def aembed_func(text: str) -> list[float]: ...

    @T.overload
    async def aembed_func(text: list[str]) -> list[list[float]]: ...

    async def aembed_func(text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            resp = await async_client.embeddings.create(input=text, model=model)
            return resp.data[0].embedding
        else:
            res = []
            for i in range(0, len(text), batch_size):
                resp = await async_client.embeddings.create(input=text[i:(i+batch_size)], model=model)
                res += [x.embedding for x in resp.data]
            return res

    if is_async:
        return aembed_func
    else:
        return embed_func


@T.overload
def create_sqlite_store_zh(
    db: str | SqliteDatabase = "",
    *,
    is_memory: bool = True,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: T.Literal[False] = False,
    echo: bool = False,
) -> tuple[SqliteDatabase, SqliteVectorStore]:
    ...


@T.overload
def create_sqlite_store_zh(
    db: str | AsyncSqliteDatabase = "",
    *,
    is_memory: bool = True,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: T.Literal[True] = True,
    echo: bool = False,
) -> tuple[AsyncSqliteDatabase, AsyncSqliteVectorStore]:
    ...


def create_sqlite_store_zh(
    db: str | SqliteDatabase | AsyncSqliteDatabase = "",
    *,
    is_memory: bool = True,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: bool = False,
    echo: bool = False,
) -> tuple[SqliteDatabase | AsyncSqliteDatabase, SqliteVectorStore | AsyncSqliteVectorStore]:
    assert not (dim is None and embedding_func is None), "You must provide dim, or embedding_func to detect dim."
    assert not (is_memory==False and not db), "You must specify database to memory or by path"
    if is_async:
        db_cls = AsyncSqliteDatabase
        vs_cls = AsyncSqliteVectorStore
        if not isinstance(db, db_cls):
            if is_memory and not db:
                db = "sqlite+aiosqlite:///:memory:"
            if not db.startswith("sqlite"):
                db = f"sqlite+aiosqlite:///{db}"
            db = db_cls(db, echo=echo)
    else:
        db_cls = SqliteDatabase
        vs_cls = SqliteVectorStore
        if not isinstance(db, db_cls):
            if is_memory and not db:
                db = "sqlite:///:memory:"
            if not db.startswith("sqlite"):
                db = f"sqlite:///{db}"
            db = db_cls(db, echo=echo)
    vs = vs_cls(
        db,
        table_prefix=store_name,
        embedding_func=embedding_func,
        dim=dim,
        clear_existed=clear_existed,
        fts_tokenize="simple",
    )
    return db, vs


@T.overload
def create_postgres_store_zh(
    db: str | PostgresDatabase = "",
    *,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: T.Literal[False] = False,
    echo: bool = False,
) -> tuple[PostgresDatabase, PostgresVectorStore]:
    ...


@T.overload
def create_postgres_store_zh(
    db: str | PostgresDatabase = "",
    *,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: T.Literal[True] = True,
    echo: bool = False,
) -> tuple[AsyncPostgresDatabase, AsyncPostgresVectorStore]:
    ...


def create_postgres_store_zh(
    db: str | PostgresDatabase | AsyncPostgresDatabase = "",
    *,
    store_name: str = "rag",
    embedding_func: T.Callable | None = None,
    dim: int | None = None,
    clear_existed: bool = False,
    is_async: bool = False,
    echo: bool = False,
) -> tuple[PostgresDatabase | AsyncPostgresDatabase, PostgresVectorStore | AsyncPostgresVectorStore]:
    if is_async:
        db_cls = AsyncPostgresDatabase
        vs_cls = AsyncPostgresVectorStore
    else:
        db_cls = PostgresDatabase
        vs_cls = PostgresVectorStore
    if isinstance(db, str):
        db = db_cls(db, echo=echo)
    vs = vs_cls(
        db,
        table_prefix=store_name,
        embedding_func=embedding_func,
        dim=dim,
        clear_existed=clear_existed,
        fts_tokenize=JiebaTokenize().as_pg_tokenize(),
    )
    return db, vs


if __name__ == "__main__":
    embed_func = create_embedding_func(base_url="http://192.168.8.68:9997/v1")
    e = embed_func("hello world")
    e = embed_func(["hello world", "hello earth"])
    print(e)
    db1, vs1 = create_sqlite_store_zh(embedding_func=embed_func)
    print(db1, vs1)
    # db2, vs2 = create_postgres_store_zh(embedding_func=embed_func)
    # print(db2, vs2)
