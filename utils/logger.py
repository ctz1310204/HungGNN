"""
TensorBoard Logger for GNN_LSAP
Copied from DL-based_LAP
"""

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    TensorBoard logger for PyTorch training
    """
    
    def __init__(self, enable_logging=True, log_dir=None):
        """
        Initialize logger
        
        Args:
            enable_logging: Whether to enable TensorBoard logging
            log_dir: Directory for TensorBoard logs (default: logs/TIMESTAMP)
        """
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            if log_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_dir = f'logs/gnn_lsap_{timestamp}'
            
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
            print(f"View with: tensorboard --logdir {log_dir}")
        else:
            self.writer = None
    
    def add_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.enable_logging:
            self.writer.add_scalar(tag, float(value), int(step))
    
    def flush(self):
        """Flush writer"""
        if self.enable_logging:
            self.writer.flush()
    
    def close(self):
        """Close writer"""
        if self.enable_logging:
            self.writer.close()
