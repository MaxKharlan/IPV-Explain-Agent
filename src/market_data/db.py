"""PostgreSQL-хранилище для market snapshot."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.market_data.moex_client import MarketSnapshot

try:
    import psycopg
except ImportError:  # pragma: no cover - dependency may be absent in scaffold mode
    psycopg = None


DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/ipv_explain_agent"


def get_database_url() -> str:
    """Возвращает DSN для подключения к PostgreSQL."""
    return os.getenv("IPV_POSTGRES_DSN", DEFAULT_DATABASE_URL)


def _require_psycopg() -> None:
    """Проверяет, что драйвер PostgreSQL доступен."""
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed. Add it to project dependencies before using PostgreSQL storage."
        )


def ensure_snapshot_table(database_url: str | None = None) -> None:
    """Создаёт таблицу хранения snapshot, если она ещё не существует."""
    _require_psycopg()
    dsn = database_url or get_database_url()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    snapshot_date DATE NOT NULL,
                    security TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (snapshot_date, security)
                )
                """
            )
        conn.commit()


def save_market_snapshot(
    snapshot: "MarketSnapshot",
    security: str = "SBER",
    database_url: str | None = None,
) -> tuple[str, str]:
    """Сохраняет MarketSnapshot в PostgreSQL."""
    _require_psycopg()
    dsn = database_url or get_database_url()
    ensure_snapshot_table(dsn)
    payload = json.dumps(asdict(snapshot), ensure_ascii=False)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_snapshots (snapshot_date, security, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (snapshot_date, security)
                DO UPDATE SET payload = EXCLUDED.payload
                """,
                (snapshot.snapshot_date, security.upper(), payload),
            )
        conn.commit()
    return snapshot.snapshot_date, security.upper()


def load_market_snapshot_by_date(
    snapshot_date: str,
    security: str = "SBER",
    database_url: str | None = None,
) -> dict[str, Any]:
    """Загружает snapshot из PostgreSQL по дате и базовому активу."""
    _require_psycopg()
    dsn = database_url or get_database_url()
    ensure_snapshot_table(dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload
                FROM market_snapshots
                WHERE snapshot_date = %s AND security = %s
                """,
                (snapshot_date, security.upper()),
            )
            row = cur.fetchone()
    if row is None:
        raise KeyError(f"No snapshot found for date={snapshot_date} security={security}")
    payload = row[0]
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def snapshot_exists(
    snapshot_date: str,
    security: str = "SBER",
    database_url: str | None = None,
) -> bool:
    """Проверяет, сохранён ли snapshot в PostgreSQL."""
    _require_psycopg()
    dsn = database_url or get_database_url()
    ensure_snapshot_table(dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM market_snapshots
                WHERE snapshot_date = %s AND security = %s
                LIMIT 1
                """,
                (snapshot_date, security.upper()),
            )
            row = cur.fetchone()
    return row is not None


def load_market_snapshots_for_period(
    t0: str,
    t1: str,
    security: str = "SBER",
    database_url: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Загружает два snapshot из PostgreSQL для дат t0 и t1."""
    snapshot_t0 = load_market_snapshot_by_date(
        t0,
        security=security,
        database_url=database_url,
    )
    snapshot_t1 = load_market_snapshot_by_date(
        t1,
        security=security,
        database_url=database_url,
    )
    return snapshot_t0, snapshot_t1


def list_stored_snapshots(
    security: str = "SBER",
    database_url: str | None = None,
) -> list[dict[str, str]]:
    """Возвращает список сохранённых snapshot по бумаге."""
    _require_psycopg()
    dsn = database_url or get_database_url()
    ensure_snapshot_table(dsn)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT snapshot_date::text, security, created_at::text
                FROM market_snapshots
                WHERE security = %s
                ORDER BY snapshot_date
                """,
                (security.upper(),),
            )
            rows = cur.fetchall()
    return [
        {
            "snapshot_date": row[0],
            "security": row[1],
            "created_at": row[2],
        }
        for row in rows
    ]
