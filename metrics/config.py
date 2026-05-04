from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetricsSettings(BaseSettings):
    """
    Configuration for metrics evaluation pipeline

    Args:
        GIGACHAT_API_KEY: API key for GigaChat
        GIGACHAT_MIN_INTERVAL_SEC: Minimum interval between API calls to GigaChat
    """

    GIGACHAT_API_KEY: str = Field(..., min_length=1)
    GIGACHAT_MIN_INTERVAL_SEC: float = Field(default=3.0, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )


metrics_settings = MetricsSettings()
