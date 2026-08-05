"""Database engine and transaction helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings


def make_engine(database_url: str | None = None):
    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(database_url), expire_on_commit=False)


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    factory = make_session_factory(database_url)
    with factory.begin() as session:
        yield session
