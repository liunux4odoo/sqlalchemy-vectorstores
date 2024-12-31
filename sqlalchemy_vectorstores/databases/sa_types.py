from __future__ import annotations

from pathlib import Path
import typing as t

import sqlalchemy as sa


DATA_PATH = Path(__file__).parent.parent / "data"


class SqliteVector(sa.TypeDecorator):
    '''
    a simple sqlalchemy column type representing embeddings in sqlite
    '''
    impl = sa.LargeBinary
    cache_ok = True

    def __init__(self, *args: t.Any, dim: int = 1024, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def process_bind_param(self, value: t.List[float | int] | None, dialect: sa.Dialect) -> bytes:
        from sqlite_vec import serialize_float32

        if value is not None:
            assert len(value) == self._dim, f"the embedding dimension({len(value)}) is not equal to ({self._dim}) in database."
            return serialize_float32(value)

    def process_result_value(self, value: bytes | None, dialect: sa.Dialect) -> t.List[float]:
        import struct

        if value is not None:
            return list(struct.unpack(f"{self._dim}f", value))

