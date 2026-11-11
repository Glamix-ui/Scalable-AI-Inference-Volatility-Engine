import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "32"))
    max_wait_ms = float(os.getenv("MAX_WAIT_MS", "8"))
    model_path = os.getenv("MODEL_PATH", "model.pt")
    allow_untrained = os.getenv("ALLOW_UNTRAINED", "0") == "1"
    torch_threads = int(os.getenv("TORCH_THREADS", "2"))
    cache_ttl = int(os.getenv("MARKET_CACHE_TTL", "120"))


settings = Settings()
