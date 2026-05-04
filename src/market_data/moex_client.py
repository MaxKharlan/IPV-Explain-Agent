"""Заготовка клиента для рыночных данных."""


def fetch_spot_prices(as_of_date: str):
    """Получает сырые спотовые цены из выбранного источника."""
    raise NotImplementedError


def fetch_yield_curve(as_of_date: str, currency: str):
    """Загружает сырые точки кривой для выбранной даты."""
    raise NotImplementedError


def fetch_option_quotes(as_of_date: str, underlier: str):
    """Загружает сырые котировки опционов для построения поверхности волатильности."""
    raise NotImplementedError
