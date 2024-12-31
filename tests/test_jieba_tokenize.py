from rich import print
from sqlalchemy_vectorstores.tokenizers.jieba_tokenize import JiebaTokenize
from sqlalchemy_vectorstores import create_sqlite_store_zh


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


def test_sqlite_fts():
    texts = [
        """NB/T20513一2018《核电厂定期安全审查指南》分为15个部分：
-一第1部分：通用要求；
一第2部分：安全性能；
第3部分：程序；
第4部分：辐射环境影响；
第5部分：概率安全分析；
第6部分：构筑物、系统和部件的实际状态；
第7部分：经验反馈；
-第8部分：老化；
-第9部分：确定论安全分析；
第10部分：人因；
第11部分：设备合格鉴定；
第12部分：设计；
第13部分：应急计划；
第14部分：灾害分析；
-第15部分：组织机构和行政管理。
本部分为NB/T20513一2018的第1部分。
本部分按照GB/T1.1一2009给出的规则起草。""",
        "本部分由能源行业核电标准化技术委员会提出。",
        """本部分起草单位：大亚湾核电运营管理有限责任公司、苏州热工研究院有限公司、中核核电运行管
理有限公司、上海核工程研究设计院有限公司。
本部分起草人：那福利、张士朋、李琪、于雪良、贺群武、刘卫华、韩镇辉。""",
        """定期安全审查的范围包括核电厂核安全的所有方面，包括核电厂运行许可证所覆盖的处在厂区内的构筑物、系统和部件及其运行，核电厂组织机构及人员配备、辐射防护、应急计划及辐射环境影响等。定期安全审查应包括以下14项安全要素：
        1a）核电厂设计（SF1）；
        b)构筑物、系统和部件的实际状态（SF2）；
        c)设备合格鉴定（SF3）；
        (P老化（SF4）；
        e)确定论安全分析（SF5）；
        f)概率安全分析（SF6）；
        g)灾害分析（SF7）；
        h)安全性能（SF8）；
        i)其他核电厂经验及研究成果的应用（SF9）；
        j)组织机构和行政管理（SF10）；
        k)程序（SF11）；
        1)人因（SF12）；
        m)应急计划（SF13）；
        n）辐射环境影响（SF14）。"""
    ]
    db, vs = create_sqlite_store_zh(is_memory=True, dim=1024, echo=True)
    for t in texts:
        vs.add_document(src_id="test", content=t)

    docs = vs.get_documents_of_source("test")
    print(docs)

    tokenize = JiebaTokenize()
    docs = vs.search_by_bm25(" OR ".join(tokenize.cut_for_search("核电厂定期安全评价的要素")))
    print(docs)
