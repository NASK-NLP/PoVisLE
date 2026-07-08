import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("stamina").setLevel(logging.ERROR)
logging.getLogger("svglib.svglib").addFilter(
    lambda record: not record.getMessage().startswith("Can't handle color: url(#")
)


def get_logger(name: str | None = None):
    return logging.getLogger(name)
