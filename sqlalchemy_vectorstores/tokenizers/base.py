from __future__ import annotations

import abc
from functools import lru_cache
import io
import re
import typing as t

import jieba
from sqlitefts import fts5


class BaseTokenize(abc.ABC): # TODO: uncompleted
    ...


@lru_cache(1)
def get_stop_words() -> t.List[str]:
    return []


def get_user_dict() -> t.List[str]:
    return []


def init_jieba():
    fp = io.StringIO()
    fp.write("\n".join(get_user_dict()))
    fp.seek(0)
    jieba.load_userdict(fp)
    fp.close()


def cut_words(text: str) -> t.Generator[t.Tuple[str, int, int], None, None]:
    '''
    jieba 分词
    '''
    stop_words = get_stop_words()
    patterns = [re.compile(r"\s+")]
    for word in jieba.cut_for_search(text):
        skip = False
        if word in stop_words:
            skip = True
        for p in patterns:
            if p.findall(word):
                skip = True
                break
        if skip:
            continue

        s = text.find(word)
        p = len(text[:s].encode())
        l = len(word.encode())
        yield word.lower(), p, l


class JiebaTokenize(fts5.FTS5Tokenizer):
    def tokenize(self, text, flags=None):
        return cut_words(text)
