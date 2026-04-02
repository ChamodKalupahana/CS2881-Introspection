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

# Create timestamped run directory
now = datetime.now()
run_name = now.strftime("run_%m_%d_%y_%H_%M")
save_root = Path(args.save_dir) / run_name
category_dirs = {}
for cat in CATEGORIES:
    d = save_root / cat
    d.mkdir(parents=True, exist_ok=True)
    category_dirs[cat] = d

# Set up logging to file + terminal
log_file = open(save_root / "run.log", "w")
sys.stdout = TeeLogger(log_file, sys.__stdout__)
sys.stderr = TeeLogger(log_file, sys.__stderr__)
