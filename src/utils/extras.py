import warnings
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

# import our own modules
# because of the setup_root in train.py and so on, we can import from src without any problems
from src.utils import get_logger, rich_utils
logger = get_logger()


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)




# def get_metric_value(metric_dict: dict, metric_name: str) -> float:
#     """Safely retrieves value of the metric logged in LightningModule."""

#     if not metric_name:
#         logger.pylogger.info("Metric name is None! Skipping metric value retrieval...")
#         return None

#     if metric_name not in metric_dict:
#         raise Exception(
#             f"Metric value not found! <metric_name={metric_name}>\n"
#             "Make sure metric name logged in LightningModule is correct!\n"
#             "Make sure `optimized_metric` name in `hparams_search` config is correct!"
#         )

#     metric_value = metric_dict[metric_name].item()
#     logger.pylogger.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

#     return metric_value
