import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from dataclasses import dataclass
import tempfile
import shutil
from ..pretrain_vilex import main, ModelArguments, DataArguments, TrainingArguments, show_memory

@pytest.fixture
def mock_cuda():
    """Mock CUDA availability and operations"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=2), \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.synchronize'), \
         patch('torch.cuda.memory_allocated', return_value=1024**3), \
         patch('torch.cuda.memory_reserved', return_value=2*1024**3), \
         patch('torch.cuda.max_memory_allocated', return_value=1.5*1024**3), \
         patch('torch.cuda.max_memory_reserved', return_value=2.5*1024**3), \
         patch('torch.cuda.current_device', return_value=0):
        yield


@pytest.fixture
def mock_distributed():
    """Mock distributed training components"""
    with patch('torch.distributed.init_process_group'), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.get_world_size', return_value=2), \
         patch('torch.distributed.barrier'), \
         patch('torch.distributed.all_reduce'), \
         patch('torch.distributed.gather_object'), \
         patch('torch.distributed.destroy_process_group'):
        yield


@pytest.fixture
def mock_wandb():
    """Mock wandb operations"""
    mock_wandb = Mock()
    mock_wandb.init.return_value = None
    mock_wandb.config = Mock()
    mock_wandb.log.return_value = None
    mock_wandb.finish.return_value = None
    with patch('wandb', mock_wandb):
        yield mock_wandb


@pytest.fixture
def mock_models():
    """Mock model components"""
    mock_llm_config = Mock()
    mock_vit_config = Mock()
    mock_vae_config = Mock()
    mock_language_model = Mock()
    mock_vit_model = Mock()
    mock_vae_model = Mock()
    mock_bagel_model = Mock()
    mock_tokenizer = Mock()
    
    # Configure mocks
    mock_language_model.init_moe.return_value = None
    mock_language_model.resize_token_embeddings.return_value = None
    mock_vit_model.vision_model.embeddings.convert_conv2d_to_linear.return_value = None
    mock_vae_model.encode.return_value = torch.randn(1, 16, 32, 32)
    mock_bagel_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    mock_bagel_model.clip_grad_norm_.return_value = torch.tensor(1.0)
    mock_tokenizer.__len__ = Mock(return_value=32000)
    
    with patch('pretrain_vilex.Qwen2Config') as mock_qwen2_config, \
         patch('pretrain_vilex.SiglipVisionConfig') as mock_siglip_config, \
         patch('pretrain_vilex.Qwen2ForCausalLM') as mock_qwen2_model, \
         patch('pretrain_vilex.SiglipVisionModel') as mock_siglip_model, \
         patch('pretrain_vilex.load_ae') as mock_load_ae, \
         patch('pretrain_vilex.Bagel') as mock_bagel, \
         patch('pretrain_vilex.Qwen2Tokenizer') as mock_qwen_tokenizer, \
         patch('pretrain_vilex.BagelConfig') as mock_bagel_config:
        
        mock_qwen2_config.from_pretrained.return_value = mock_llm_config
        mock_qwen2_config.from_json_file.return_value = mock_llm_config
        mock_siglip_config.from_pretrained.return_value = mock_vit_config
        mock_siglip_config.from_json_file.return_value = mock_vit_config
        mock_qwen2_model.from_pretrained.return_value = mock_language_model
        mock_qwen2_model.return_value = mock_language_model
        mock_siglip_model.from_pretrained.return_value = mock_vit_model
        mock_siglip_model.return_value = mock_vit_model
        mock_load_ae.return_value = (mock_vae_model, mock_vae_config)
        mock_bagel.return_value = mock_bagel_model
        mock_qwen_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        yield {
            'llm_config': mock_llm_config,
            'vit_config': mock_vit_config,
            'vae_config': mock_vae_config,
            'language_model': mock_language_model,
            'vit_model': mock_vit_model,
            'vae_model': mock_vae_model,
            'bagel_model': mock_bagel_model,
            'tokenizer': mock_tokenizer
        }


@pytest.fixture
def mock_training_components():
    """Mock training-related components"""
    mock_optimizer = Mock()
    mock_scheduler = Mock()
    mock_loader = Mock()
    mock_fsdp_model = Mock()
    mock_ema_model = Mock()
    
    # Configure mocks
    mock_optimizer.param_groups = [{'lr': 1e-4}]
    mock_optimizer.zero_grad.return_value = None
    mock_optimizer.step.return_value = None
    mock_scheduler.step.return_value = None
    mock_fsdp_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    mock_fsdp_model.clip_grad_norm_.return_value = torch.tensor(1.0)
    mock_fsdp_model.train.return_value = None
    mock_ema_model.eval.return_value = None
    
    # Mock data loader with single batch
    mock_data = {
        'padded_images': torch.randn(2, 3, 512, 512),
        'ce_loss_indexes': [0, 1, 2],
        'mse_loss_indexes': [0, 1, 2],
        'sample_lens': [100, 150],
        'batch_data_indexes': [{'dataset_name': 'test', 'worker_id': 0, 'data_indexes': [0, 1]}]
    }
    mock_loader.__iter__ = Mock(return_value=iter([mock_data]))
    
    # Mock loss dict
    mock_loss_dict = {
        'ce': torch.tensor([0.5, 0.6, 0.4]),
        'mse': torch.randn(3, 16)
    }
    mock_fsdp_model.return_value = mock_loss_dict
    
    with patch('torch.optim.AdamW', return_value=mock_optimizer), \
         patch('pretrain_vilex.get_constant_schedule_with_warmup', return_value=mock_scheduler), \
         patch('pretrain_vilex.get_cosine_with_min_lr_schedule_with_warmup', return_value=mock_scheduler), \
         patch('pretrain_vilex.get_loader', return_value=mock_loader), \
         patch('pretrain_vilex.fsdp_wrapper', return_value=mock_fsdp_model), \
         patch('pretrain_vilex.fsdp_ema_setup', return_value=mock_ema_model), \
         patch('pretrain_vilex.apply_activation_checkpointing'), \
         patch('pretrain_vilex.fsdp_ema_update'), \
         patch('copy.deepcopy', return_value=mock_ema_model):
        yield {
            'optimizer': mock_optimizer,
            'scheduler': mock_scheduler,
            'loader': mock_loader,
            'fsdp_model': mock_fsdp_model,
            'ema_model': mock_ema_model
        }


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing"""
    temp_dir = tempfile.mkdtemp()
    results_dir = os.path.join(temp_dir, 'results')
    checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
    model_dir = os.path.join(temp_dir, 'model')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    yield {
        'temp_dir': temp_dir,
        'results_dir': results_dir,
        'checkpoint_dir': checkpoint_dir,
        'model_dir': model_dir
    }
    
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_args():
    """Mock command line arguments"""
    with patch('sys.argv', ['pretrain_vilex.py']):
        yield


class TestShowMemory:
    """Test the show_memory utility function"""
    
    def test_show_memory_with_logger(self, mock_cuda):
        mock_logger = Mock()
        show_memory("TEST", mock_logger)
        mock_logger.info.assert_called_once()
        
    def test_show_memory_without_logger(self, mock_cuda, capsys):
        show_memory("TEST")
        captured = capsys.readouterr()
        assert "MEMORY TEST" in captured.out
        assert "Allocated:" in captured.out
        assert "Reserved:" in captured.out


class TestMain:
    """Test the main training function"""
    
    def test_main_basic_execution(self, mock_cuda, mock_distributed, mock_wandb, 
                                 mock_models, mock_training_components, temp_dirs, mock_args):
        """Test basic main function execution without errors"""
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10, 20]):  # Mock time progression
            
            # Mock argument parsing
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.total_steps = 5  # Short training for test
            training_args.log_every = 1
            training_args.save_every = 10  # Avoid saving in short test
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            # Mock logger
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            
            # Mock special tokens
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            
            # Mock checkpoint operations
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                0, None
            )
            
            # Run main function
            main()
            
            # Verify key operations were called
            mock_logger.info.assert_called()
            mock_wandb.init.assert_called_once()
            
    def test_main_finetune_from_hf(self, mock_cuda, mock_distributed, mock_wandb,
                                  mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function with finetune_from_hf=True"""
        
        # Create mock config files
        llm_config_path = os.path.join(temp_dirs['model_dir'], 'llm_config.json')
        vit_config_path = os.path.join(temp_dirs['model_dir'], 'vit_config.json')
        ae_path = os.path.join(temp_dirs['model_dir'], 'ae.safetensors')
        
        with open(llm_config_path, 'w') as f:
            f.write('{"hidden_size": 768}')
        with open(vit_config_path, 'w') as f:
            f.write('{"hidden_size": 768}')
        with open(ae_path, 'w') as f:
            f.write('mock_ae_data')
            
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10]):
            
            # Configure for HF finetuning
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.finetune_from_hf = True
            training_args.total_steps = 1
            training_args.log_every = 1
            training_args.save_every = 10
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                0, None
            )
            
            main()
            
            # Verify HF-specific path was used
            mock_logger.info.assert_called()
            
    def test_main_auto_resume(self, mock_cuda, mock_distributed, mock_wandb,
                            mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function with auto resume functionality"""
        
        mock_resume_path = "/fake/resume/path"
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=mock_resume_path), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10]):
            
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.auto_resume = True
            training_args.total_steps = 1
            training_args.log_every = 1
            training_args.save_every = 10
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                100, None  # Resume from step 100
            )
            
            main()
            
            # Verify resume path was used
            mock_checkpoint.try_load_ckpt.assert_called_with(
                mock_resume_path, mock_logger, mock_models['bagel_model'], 
                mock_training_components['ema_model'], resume_from_ema=False
            )
            
    def test_main_cosine_scheduler(self, mock_cuda, mock_distributed, mock_wandb,
                                 mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function with cosine learning rate scheduler"""
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10]):
            
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.lr_scheduler = 'cosine'  # Use cosine scheduler
            training_args.total_steps = 1
            training_args.log_every = 1
            training_args.save_every = 10
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                0, None
            )
            
            main()
            
            mock_logger.info.assert_called()
            
    def test_main_invalid_scheduler(self, mock_cuda, mock_distributed, mock_wandb,
                                   mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function with invalid learning rate scheduler"""
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'):
            
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.lr_scheduler = 'invalid'  # Invalid scheduler
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            
            # Should raise ValueError for invalid scheduler
            with pytest.raises(ValueError):
                main()
                
    def test_main_no_cuda(self):
        """Test main function fails when CUDA is not available"""
        
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(AssertionError):
                main()
                
    def test_main_freeze_components(self, mock_cuda, mock_distributed, mock_wandb,
                                  mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function with component freezing enabled"""
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10]):
            
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.freeze_llm = True
            training_args.freeze_vit = True
            training_args.freeze_vae = True
            training_args.total_steps = 1
            training_args.log_every = 1
            training_args.save_every = 10
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 0)
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                0, None
            )
            
            main()
            
            # Verify eval() was called for frozen components
            mock_models['language_model'].eval.assert_called_once()
            mock_models['vit_model'].eval.assert_called_once()
            
    def test_main_new_tokens(self, mock_cuda, mock_distributed, mock_wandb,
                           mock_models, mock_training_components, temp_dirs, mock_args):
        """Test main function when new tokens are added to tokenizer"""
        
        with patch('pretrain_vilex.HfArgumentParser') as mock_parser, \
             patch('pretrain_vilex.create_logger') as mock_create_logger, \
             patch('pretrain_vilex.add_special_tokens') as mock_add_tokens, \
             patch('pretrain_vilex.get_latest_ckpt', return_value=None), \
             patch('pretrain_vilex.FSDPCheckpoint') as mock_checkpoint, \
             patch('pretrain_vilex.set_seed'), \
             patch('time.time', side_effect=[0, 10]):
            
            model_args = ModelArguments()
            model_args.model_path = temp_dirs['model_dir']
            data_args = DataArguments()
            training_args = TrainingArguments()
            training_args.results_dir = temp_dirs['results_dir']
            training_args.checkpoint_dir = temp_dirs['checkpoint_dir']
            training_args.total_steps = 1
            training_args.log_every = 1
            training_args.save_every = 10
            
            mock_parser.return_value.parse_args_into_dataclasses.return_value = (
                model_args, data_args, training_args
            )
            
            mock_logger = Mock()
            mock_create_logger.return_value = mock_logger
            # Mock adding new tokens
            mock_add_tokens.return_value = (mock_models['tokenizer'], {}, 5)  # 5 new tokens
            mock_checkpoint.try_load_ckpt.return_value = (
                mock_models['bagel_model'], mock_training_components['ema_model']
            )
            mock_checkpoint.try_load_train_state.return_value = (
                mock_training_components['optimizer'], 
                mock_training_components['scheduler'], 
                0, None
            )
            
            main()
            
            # Verify token embeddings were resized
            mock_models['language_model'].resize_token_embeddings.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])