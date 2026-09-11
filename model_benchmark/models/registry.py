"""Single place every other phase imports the model zoo from.

Every entry exposes the same two-function interface:
    fit(series) -> object          # series = (years, values)
    predict(fitted, h) -> array of length h, on the population scale
"""

from models import arima, ets, exponential, linear_trend, modified_exponential, naive
from models.common import MIN_OBS

MODELS = {
    "naive": naive,
    "linear_trend": linear_trend,
    "exponential": exponential,
    "modified_exponential": modified_exponential,
    "ets_damped": ets,
    "arima_111": arima,
}


def fit_predict(name, series, h):
    """Convenience wrapper: fit model `name` on `series` and return an
    h-length forecast, or None if the model failed/had too few observations."""
    years, values = series
    if len(values) < MIN_OBS[name]:
        return None
    try:
        fitted = MODELS[name].fit(series)
        return MODELS[name].predict(fitted, h)
    except Exception:
        return None
