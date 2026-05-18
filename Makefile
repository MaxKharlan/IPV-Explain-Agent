PYTHON ?= python3
SECURITY ?= SBER
DATE ?= 2025-05-05
T0 ?= 2025-04-28
T1 ?= 2025-05-05

.PHONY: postgres-up postgres-down postgres-logs snapshot-build snapshot-period snapshot-list

postgres-up:
	docker compose up -d postgres

postgres-down:
	docker compose down

postgres-logs:
	docker compose logs -f postgres

snapshot-build:
	$(PYTHON) -c "from src.market_data.moex_client import build_and_store_market_snapshot; s = build_and_store_market_snapshot('$(DATE)', security='$(SECURITY)'); print(s.snapshot_id, s.source)"

snapshot-period:
	$(PYTHON) -c "from src.market_data.moex_client import build_and_store_market_snapshots_for_period; s0, s1 = build_and_store_market_snapshots_for_period('$(T0)', '$(T1)', security='$(SECURITY)'); print(s0.snapshot_id, s0.source); print(s1.snapshot_id, s1.source)"

snapshot-list:
	$(PYTHON) -c "from src.market_data.db import list_stored_snapshots; print(list_stored_snapshots(security='$(SECURITY)'))"
