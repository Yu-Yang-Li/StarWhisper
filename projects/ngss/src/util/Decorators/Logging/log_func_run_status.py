from functools import wraps
from typing import Any, Callable

from loguru import logger
from loguru._logger import Logger


class LogFuncRun:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        log_note_dict: dict = kwargs["log_note"]
        logger: Logger = kwargs["debug_logger"]

        log_note: list[str] = [
            f"{item[0]}: {item[1]}" for item in log_note_dict.items()
        ]
        logger.trace(
            f"Begin to run {self.func.__name__}. Log note: {', '.join(log_note)}",
        )

        try:
            result = self.func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to run {self.func.__name__}" + str(e))
            raise e
        else:
            logger.trace(f"Finished running {self.func.__name__}")
            return result


def log_func_run(logger: Logger, note=None):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.trace(f"Begin to run {func.__name__}. Note: {note}")
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to run {func.__name__}" + str(e))
            else:
                logger.trace(f"Finished running {func.__name__}")
                return result

        return wrapper

    return decorator
