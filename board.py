from pathlib import Path
import tensorflow as tf
import datetime as dt

BASE_LOG_DIR = Path(__file__).resolve().parent / 'tensor-logs'

def get_tensorboard_callback(subdir: str = "fit", with_timestamp: bool = True) -> tf.keras.callbacks.TensorBoard:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S") if with_timestamp else ""
    log_dir = BASE_LOG_DIR / subdir / stamp if stamp else BASE_LOG_DIR / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        profile_batch=0,
    )