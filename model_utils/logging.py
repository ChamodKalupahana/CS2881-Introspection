import sys
from datetime import datetime
from pathlib import Path

class TeeLogger:
    """Duplicate writes to both a file and the original stream."""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

def setup_logging(base_save_dir: str, categories: list = None, run_name: str = None):
    """
    Sets up a timestamped run directory and redirects stdout/stderr to a log file.
    
    Args:
        base_save_dir: The root directory for all runs.
        categories: Optional list of category subdirectories to create.
        
    Returns:
        save_root (Path): The Path to the current run's directory.
        log_file (file object): The opened log file (need to close it at the end).
    """
    if not run_name:
        now = datetime.now()
        run_name = now.strftime("run_%m_%d_%y_%H_%M")
    
    save_root = Path(base_save_dir) / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    if categories:
        for cat in categories:
            (save_root / cat).mkdir(parents=True, exist_ok=True)

    log_path = save_root / "run.log"
    log_file = open(log_path, "w")
    
    # Redirect stdout and stderr
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)
    
    print(f"📝 Logging initialized. Run directory: {save_root}")
    return save_root, log_file
