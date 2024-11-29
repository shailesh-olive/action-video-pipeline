from pathlib import Path

from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.yaml", ".secrets.yaml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.


settings.BASE_DIR = Path(__file__).parent.parent.absolute()
settings.OUTPUT_DIR = settings.BASE_DIR / "artifacts/outputs"
settings.LOG_DIR = settings.OUTPUT_DIR / "logs"

settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

# logger.add(
#     settings.LOG_DIR / "debug.log",
#     filter=lambda record: record["level"].name == "DEBUG",
# )
# logger.add(
#     settings.LOG_DIR / "info.log",
#     filter=lambda record: record["level"].name == "INFO",
# )
# logger.add(
#     settings.LOG_DIR / "error.log",
#     filter=lambda record: record["level"].name == "ERROR",
# )
#
# logger.add(
#     settings.LOG_DIR / "warn.log",
#     filter=lambda record: record["level"].name == "WARNING",
# )
