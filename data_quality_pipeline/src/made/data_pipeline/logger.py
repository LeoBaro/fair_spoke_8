
import logging

def setup_logger(worker_id):
    logger = logging.getLogger(f"worker_{worker_id}")
    handler = logging.FileHandler(f"worker_{worker_id}_log.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_filtering_stats(worker_id, batch, num_filtered, elapsed_time):
    logger = setup_logger(worker_id)
    logger.info("Batch processed: %s, Filtered items: %s, Execution time: %s seconds", batch, num_filtered, elapsed_time)
