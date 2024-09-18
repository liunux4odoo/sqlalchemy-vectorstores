from __future__ import annotations

import abc
import typing as t


class BaseTokenize(abc.ABC):
    '''
    An unit class to cut text to fts tokens for sqlite & postgres
    '''
    def __init__(self, stop_words: t.List[str]=[], user_dict: t.List[t.Dict]=[]) -> None:
        self._sqlite_tokenize = None
        self._pg_tokenize = None
        self._stop_words = stop_words
        self._user_dict = user_dict

    def load_stop_words(self) -> t.List[str]:
        return self._stop_words
    
    def load_user_dict(self) -> t.List[t.Dict]:
        return self._user_dict

    @abc.abstractmethod
    def cut_words(self, text: str) -> t.Generator[t.Tuple[str, int, int], None, None]:
        '''
        cut sentence to words, the return value must be: (word, start_pos, end_pos)
        '''
        ...

    def as_sqlite_tokenize(self):
        if self._sqlite_tokenize is None:
            from sqlitefts import fts5
            class CustomTokenize(fts5.FTS5Tokenizer):
                def tokenize(that, text, flags=None):
                    for t,s,e in self.cut_words(text):
                        s = len(text[:s].encode("utf-8"))
                        e = s + len(t.encode("utf-8"))
                        yield t, s, e
            self._sqlite_tokenize = fts5.make_fts5_tokenizer(CustomTokenize())
        return self._sqlite_tokenize

    def as_pg_tokenize(self):
        if self._pg_tokenize is None:
            def tokenize(text: str) -> str:
                res = {}
                for t,s,e in self.cut_words(text):
                    if t not in res:
                        res[t] = f"'{t}':{s+1}"
                    else:
                        res[t] = res[t] + f",{s+1}"
                return " ".join(res.values())
            self._pg_tokenize = tokenize
        return self._pg_tokenize
