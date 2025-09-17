"""
Memory-efficient training evaluation module - No text-to-image generation

This module provides evaluation utilities without memory-intensive operations:
1. Parameter freeze status logging
2. First batch debugging and visualization  
3. Attention map visualization
4. Batch analysis and statistics
"""
import torch
from training_debug import create_training_debugger
import torch.distributed as dist
import os


class MinimalTrainingEvaluationSuite:
    """Memory-efficient training evaluation suite - no text-to-image generation"""
    
    def __init__(self, model, tokenizer, output_dir, run_name="run01"):
        """Initialize with minimal components (no VAE model for generation)"""
        self.model = model
        self.output_dir = output_dir
        self.run_name = run_name
        
        # Initialize debugger only
        self.debugger = create_training_debugger(output_dir, run_name)
        
        # Track if first batch has been processed
        self._first_batch_processed = False
    
    def process_training_start(self, logger, step=0):
        """Process training start - log parameter status only (no generation)"""
        try:
            # Only on main process
            if torch.distributed.is_initialized() and dist.get_rank() != 0:
                return
                
            # Log parameter freeze status
            self.debugger.log_parameter_freeze_status(self.model, logger, step)
            
            # Log memory status
            if hasattr(logger, 'info'):
                logger.info(f"Training evaluation initialized - memory-efficient mode (no text-to-image generation)")
            else:
                print(f"Training evaluation initialized - memory-efficient mode (no text-to-image generation)")
            
        except Exception as e:
            if hasattr(logger, 'error'):
                logger.error(f"Error in training start processing: {e}")
            else:
                print(f"Error in training start processing: {e}")
    
    def process_first_batch(self, batch, tokenizer, logger, step=0):
        """Process first training batch - comprehensive analysis and visualization"""
        try:
            if self._first_batch_processed:
                return  # Only process once
                
            # Only on main process
            if torch.distributed.is_initialized() and dist.get_rank() != 0:
                self._first_batch_processed = True
                return
            
            # Log and save first batch data
            self.debugger.log_and_save_first_batch(batch, tokenizer, logger, step)
            
            # Save attention map
            attention_path = self.debugger.save_attention_map(batch, step)
            if attention_path:
                if hasattr(logger, 'info'):
                    logger.info(f"Attention map saved to: {attention_path}")
                else:
                    print(f"Attention map saved to: {attention_path}")
            
            self._first_batch_processed = True
            
        except Exception as e:
            if hasattr(logger, 'error'):
                logger.error(f"Error in first batch processing: {e}")
            else:
                print(f"Error in first batch processing: {e}")
    
    def process_checkpoint_save(self, step, logger=None):
        """Process checkpoint save - log status only (no generation)"""
        try:
            # Only on main process
            if torch.distributed.is_initialized() and dist.get_rank() != 0:
                return
                
            # Just log the checkpoint save event
            if logger and hasattr(logger, 'info'):
                logger.info(f"Checkpoint saved at step {step} - evaluation skipped (memory-efficient mode)")
            else:
                print(f"Checkpoint saved at step {step} - evaluation skipped (memory-efficient mode)")
                
        except Exception as e:
            if logger and hasattr(logger, 'error'):
                logger.error(f"Error in checkpoint save processing: {e}")
            else:
                print(f"Error in checkpoint save processing: {e}")
    
    def log_training_statistics(self, step, loss_dict, logger=None):
        """Log training statistics and batch information"""
        try:
            # Only on main process
            if torch.distributed.is_initialized() and dist.get_rank() != 0:
                return
                
            # Log current training status
            message = f"Step {step} - "
            for key, value in loss_dict.items():
                if hasattr(value, 'item'):
                    message += f"{key}: {value.item():.4f}, "
                else:
                    message += f"{key}: {value:.4f}, "
            
            if logger and hasattr(logger, 'info'):
                logger.info(message)
            else:
                print(message)
                
        except Exception as e:
            if logger and hasattr(logger, 'error'):
                logger.error(f"Error logging training statistics: {e}")
            else:
                print(f"Error logging training statistics: {e}")


def create_minimal_training_evaluation_suite(model, tokenizer, output_dir, run_name="run01"):
    """Factory function to create memory-efficient training evaluation suite"""
    return MinimalTrainingEvaluationSuite(model, tokenizer, output_dir, run_name)
