import logging
import os
import sys
from datetime import datetime

def setup_logger(name, save_dir=None, filename="experiment.log", level=logging.INFO):
    """
    Sets up a logger that outputs to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class ExperimentLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logger = setup_logger("Experiment", save_dir=save_dir)
        # Placeholder for Tensorboard writer if needed
        # self.writer = SummaryWriter(log_dir=save_dir)

    def log_metrics(self, step, metrics):
        """
        Log dictionary of metrics.
        """
        msg = f"Step {step}: " + ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(msg)
        # if self.writer:
        #     for k, v in metrics.items():
        #         self.writer.add_scalar(k, v, step)

    def log_text(self, step, text):
        self.logger.info(f"Step {step} [TEXT]: {text}")

    def close(self):
        # if self.writer:
        #     self.writer.close()
        pass
