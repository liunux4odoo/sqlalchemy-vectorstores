from __future__ import annotations

import re
import typing as t

from .base import BaseTokenize


class JiebaTokenize(BaseTokenize):
    def __init__(
        self,
        stop_words: t.List[str] = [],
        user_dict: t.List[str, t.Tuple, t.Dict] = [],
    ) -> None:
        super().__init__(stop_words=stop_words, user_dict=user_dict)

        import jieba
        jieba.initialize()
        for w in self.load_user_dict():
            word = None
            freq = None
            tag = None

            if isinstance(w, str):
                word = w
            elif isinstance(w, (tuple, list)):
                if len(w) >= 1:
                    word = w[0]
                if len(w) >= 2:
                    freq = w[1]
                if len(w) >= 3:
                    tag = w[2]
            elif isinstance(w, dict):
                word = w.get("word")
                freq = w.get("freq")
                tag = w.get("tag")
            jieba.add_word(word, freq, tag)

    def cut_words(
        self,
        text: str,
        skip_patterns: t.List[re.Pattern]=[re.compile(r"\s+")],
    ) -> t.Generator[t.Tuple[str, int, int], None, None]:
        '''
        jieba 分词
        '''
        import jieba

        stop_words = self.load_stop_words()
        for word,s,e in jieba.tokenize(text):
            skip = False
            if word in stop_words or word.lower() in stop_words:
                skip = True
            for p in skip_patterns:
                if p.match(word):
                    skip = True
                    break
            if skip:
                continue
            yield word.lower(), s, e

    def cut_for_search(self, text: str) -> list[str]:
        import jieba

        return [x.lower() for x in jieba.lcut_for_search(text) if x.strip()]
