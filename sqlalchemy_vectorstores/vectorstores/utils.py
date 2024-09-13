from __future__ import annotations

import sqlalchemy as sa


def _select_first_to_dict(result: sa.ResultProxy) -> dict | None:
    '''
    helper function to convert select query to dict
    '''
    if data := result.first():
        return {k:v for k,v in zip(result.keys(), data)}
