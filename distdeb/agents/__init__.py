from .base import Agent, Forecast
from .arima_agent import ARIMAAgent


def __getattr__(name):
    # Lazy imports: don't pull in torch / timesfm / uni2ts / transformers
    # unless that specific agent is requested.
    if name == "ChronosBoltAgent":
        from .chronos_agent import ChronosBoltAgent
        return ChronosBoltAgent
    if name == "TimesFMAgent":
        from .timesfm_agent import TimesFMAgent
        return TimesFMAgent
    if name == "MoiraiAgent":
        from .moirai_agent import MoiraiAgent
        return MoiraiAgent
    if name == "QwenLLMTimeAgent":
        from .llmtime_agent import QwenLLMTimeAgent
        return QwenLLMTimeAgent
    raise AttributeError(name)


__all__ = [
    "Agent",
    "Forecast",
    "ARIMAAgent",
    "ChronosBoltAgent",
    "TimesFMAgent",
    "MoiraiAgent",
    "QwenLLMTimeAgent",
]
