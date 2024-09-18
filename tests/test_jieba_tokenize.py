from rich import print
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize


def test_tokenize():
    tokenize = JiebaTokenize(user_dict=["sqlalchemy-vectores"])
    text = "sqlalchemy-vectores 是一个通过 sqlalchemy 利用 sqlite 和 postgres 数据库实现向量检索和 BM25 全文检索功能的库。"

    # basic words cut
    print("basic words cut")
    r = list(tokenize.cut_words(text))
    print(r)
    assert "sqlalchemy-vectores" in [x[0] for x in r]

    # # sqlite fts tokenize
    print("sqlite fts tokenize")
    r = tokenize.as_sqlite_tokenize()
    print(r)

    # postgres to_tsvector tokenize
    print("postgres to_tsvector tokenize")
    r = tokenize.as_pg_tokenize()(text)
    print(r)
    assert "'sqlalchemy-vectores':1" in r
