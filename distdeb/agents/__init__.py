from .base import Agent, Forecast
from .arima_agent import ARIMAAgent

# Lazy-export ChronosBoltAgent so importing the package doesn't require torch.
def __getattr__(name):
    if name == "ChronosBoltAgent":
        from .chronos_agent import ChronosBoltAgent
        return ChronosBoltAgent
    raise AttributeError(name)

__all__ = ["Agent", "Forecast", "ARIMAAgent", "ChronosBoltAgent"]
