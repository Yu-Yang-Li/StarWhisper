from functools import wraps
from typing import Any, Callable

from icecream import ic
from loguru import logger

ic.enable()


class LogIterProgress:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        accu_iteration = kwargs["accu_iteration"]
        total_iteration = kwargs["total_iteration"]
        ic(self.func.__name__)
        result = self.func(*args, **kwargs)
        if kwargs["accu_iteration"] % kwargs["report_interval"] == 0:
            logger.info(
                f"{str(accu_iteration)} out of {total_iteration} iterations is finished for {self.func.__name__}"
            )

        return result


def log_iteration_progress(accu_iteration, total_iteration, report_interval):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            if accu_iteration % report_interval == 0:
                logger.info(
                    f"{str(accu_iteration)} out of {total_iteration} iterations is finished for {func.__name__}"
                )

        return wrapper

    return decorator
