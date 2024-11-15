from __future__ import annotations

import enum
import typing as t

import sqlalchemy as sa


def _select_first_to_dict(result: sa.ResultProxy) -> dict | None:
    '''
    helper function to convert select query to dict
    '''
    if data := result.first():
        return {k:v for k,v in zip(result.keys(), data)}


class Document(t.TypedDict):
    src_id: str
    content: str
    metadata: dict
    type: str | None
    target_id: str | None


class DocType(str, enum.Enum):
    ORIGIN = "origin"
    SUMMARY = "summary"
    QUESTION = "question"
    ANSWER = "answer"
    DESCRIPTION = "description"
    OTHER = "other"
