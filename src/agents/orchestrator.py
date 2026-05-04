"""Скелет интерфейса для пайплайна агента."""


def get_position(position_id: str):
    """Возвращает нормализованные данные о позиции."""
    raise NotImplementedError


def load_market_snapshot(snapshot_date: str, instrument_context: dict):
    """Возвращает нормализованную картину рынка за определённую дату."""
    raise NotImplementedError


def compute_fair_value(position, market_t0, market_t1):
    """Результаты модели, необходимые для атрибуции."""
    raise NotImplementedError


def run_attribution(pricing_result):
    """Возвращает готовую к объяснению разбивку прибыли и убытка."""
    raise NotImplementedError


def generate_narrative(attribution):
    """Возвращает проверенный текстовый LLM вывод."""
    raise NotImplementedError


def build_report(position, attribution, narrative):
    """Возвращает payload отчета для экспорта в формат PDF или HTML."""
    raise NotImplementedError
