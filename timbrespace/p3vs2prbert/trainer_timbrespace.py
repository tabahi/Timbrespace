import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import Namespace
import time


from loss import MaskedLanguageModelLoss
from p3vs2prbert.model_timbrespace import ProsodicSequenceEncoder



class SequenceEncoderPipeline(L.LightningModule):
    
    def __init__(self, hp=None):
        super(SequenceEncoderPipeline, self).__init__()
        
        self.model_ready = False
        # Only set hyperparameters if explicitly provided during initialization
        if hp is not None:
            if (type(hp) == dict):
                hp = Namespace(**hp)
            self.hp = hp
            self.exp_name = "default"
            self.save_hyperparameters(self.hp)
            # Note: we'll build the model in the configure method instead
        
        self.cupe_ckpt_path = None
        self.model = None
        self.main_node = True
        self.last_epoch_time = time.time()
        self.is_pre_trained = False

    
    
    def update_hyperparams(self, hp):
        '''
        Mainly needed to update the hyperparameters after loading from checkpoint. Mainly for the loss function penalty values.
        '''
        self.hp = hp
        self.save_hyperparameters(self.hp)
        self.last_epoch_time = time.time()


    def configure(self, cupe_ckpt_path=None, main_node=True):
        """Configure the model, called from the main script before training."""
        self.main_node = main_node

        
        
        if not self.model_ready:
            self.build_model()
            
        return self.model_ready
    
        

    def build_model(self):
        print(".............................building model")
        
        self.mask_prob_now = self.hp.mask_prob_init
        self.mask_prob_init = self.hp.mask_prob_init
        self.mask_prob_max = self.hp.mask_prob_max  


        # Clear vocabulary management
        self.base_vocab_size = self.hp.phoneme_vocab_size
        if 'phoneme_noise_token' in self.hp and (self.hp.phoneme_noise_token is not None):
            blank_token_id = self.hp.phoneme_noise_token
        else:
            blank_token_id = self.base_vocab_size  # Use next available ID
            self.base_vocab_size += 1  # Increment to include blank token
        
        self.model = ProsodicSequenceEncoder(
            pos_max_len=self.hp.pos_max_len,
            hidden_dim=self.hp.hidden_dim,
            num_layers=self.hp.num_layers,
            vocab_size=self.base_vocab_size,  # Pass the base vocabulary size
            dropout_rate=self.hp.dropout_rate,
            mask_prob=self.mask_prob_now,
            padding_token_id=-100
        )
    
        self.loss_fn = MaskedLanguageModelLoss(
            blank_token_id=blank_token_id,
            vocab_size=self.base_vocab_size,  # Match prediction head output
            padding_token_id=self.model.padding_token_id,  # Use model's internal padding ID
            blank_penalty_weight=self.hp.blank_penalty_weight,
        )

        self.hp.base_vocab_size = self.base_vocab_size  # Store base vocab size in hyperparameters
        self.save_hyperparameters(self.hp)
        self.exp_name = self.hp.experiment + "_" + self.hp.model_version
        self.params_count = sum(p.numel() for p in self.parameters())
        print(".............................model built.")
        self.model_ready = True
        return self.model_ready
    
         
    def on_load_checkpoint(self, checkpoint):
        """Customize checkpoint loading to handle model structure changes between versions"""
        # Extract hyperparameters from the checkpoint
        
        hp = checkpoint['hyper_parameters']
        
        if (type(hp) == dict):
            hp = Namespace(**hp)
        self.hp = hp
        
        print("Loaded hp from checkpoint")
        
        #self.ph_seq_max = self.hp.ph_seq_max
        # Build the model with the restored hyperparameters
        # This ensures the model structure is created before loading weights
        self.build_model()

        
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = {
                k: v for k, v in checkpoint['state_dict'].items()
                if (not k.startswith('loss_fn') and not k.startswith('model_config'))
            }
        
        self.is_pre_trained = True
        print(f"Checkpoint adapted: {len(checkpoint['state_dict'])} parameters will be loaded")
        
    
    
    def update_hyperparams(self, hp):
        '''
        Mainly needed to update the hyperparameters after loading from checkpoint. Mainly for the loss function penalty values.
        '''
        self.hp = hp
        self.save_hyperparameters(self.hp)
        self.last_epoch_time = time.time()
    
       


    
    
        
            
        
    def forward(self, data_batch, batch_idx, mode):
        # Unpack the data batch
        # Note: ph_seqs should now contain K-means label sequences instead of continuous features
        ph_seqs, ph_seq_lens, _c, _r = data_batch
        
        # ph_seqs should be [batch_size, seq_len] with integer token IDs
        
        # Ensure ph_seqs are integers
        if ph_seqs.dtype != torch.long:
            raise ValueError("ph_seqs must be of type torch.long (integer token IDs)")
        
        
        
        # Handle prediction mode differently
        if mode == 'pred':
            sequence_embedding = self.model(ph_seqs, lengths=ph_seq_lens, training=False)
            return sequence_embedding
        
        
        # For training, model returns: logits, transformer_output, target_mask, original_tokens
        logits, transformer_output, target_mask, original_tokens = self.model(ph_seqs, lengths=ph_seq_lens, training=True)
        
        
        # Validate outputs
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"NaN or Inf detected in logits at batch {batch_idx}")
            logits = torch.where(
                torch.isnan(logits) | torch.isinf(logits),
                torch.zeros_like(logits),
                logits
            )

        if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
            print(f"NaN or Inf detected in transformer output at batch {batch_idx}")
            transformer_output = torch.where(
                torch.isnan(transformer_output) | torch.isinf(transformer_output),
                torch.zeros_like(transformer_output),
                transformer_output
            )

        model_output = (logits, transformer_output, target_mask, original_tokens, ph_seq_lens)
        loss, loss_components, debug_info = self.loss_fn(model_output, training=(mode == 'train'))
        
        
        # Log metrics
        self.log(f'{mode}_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log detailed loss components
        if mode in ['val', 'test', 'train']:
            for key, value in loss_components.items():
                self.log(f'{mode}_{key}', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            
            # Also log masking percentage for monitoring
            mask_percentage = 100 * target_mask.float().mean().item()
            self.log(f'{mode}_mask_percentage', mask_percentage, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            
            # Log token prediction accuracy on masked positions
            if mode in ['val', 'test']:
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions = (predictions == original_tokens) & target_mask
                    accuracy = correct_predictions.sum().float() / target_mask.sum().float()
                    self.log(f'{mode}_token_accuracy', accuracy, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

                    if (accuracy>0.1) and (mode == 'val') and (batch_idx % 10 == 0) and self.current_epoch > 0:
                        print(f"Batch {batch_idx} - {mode} token accuracy: {accuracy:.4f}")
                        print(predictions[0][:50])
                        print(predictions[-1][:50]) 

                        print(debug_info)
        
        return loss


    

    def training_step(self, data_batch, batch_idx):
        """Training step"""
        return self.forward(data_batch, batch_idx, 'train')

    def validation_step(self, data_batch, batch_idx):
        """Validation step"""
        return self.forward(data_batch, batch_idx, 'val')

    def test_step(self, data_batch, batch_idx):
        """Test step"""
        return self.forward(data_batch, batch_idx, 'test')
    
    def predict_step(self, data_batch, batch_idx):
        
        """predict step"""
        return self.forward(data_batch, batch_idx, 'pred')
    
    
    def on_test_epoch_end(self):
        """Test epoch end callback"""
        print("on_test_epoch_end")
        #return self.on_end_metrics_eval('test')
    
    def on_test_end(self):
        """Test end callback"""
        print("on_test_end")
        return {}


    def on_train_epoch_end(self):
        """Enhanced training epoch end with smart adaptations"""
        
        epoch_time_minutes = round((time.time() - self.last_epoch_time) / 60, 1)
        self.log('et', epoch_time_minutes, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.last_epoch_time = time.time()

        # Get enhanced analysis from loss function
        mask_accuracy, recommendations = self.loss_fn.on_train_epoch_end(
            self.current_epoch, 
            self.trainer.max_epochs
        )
        
        self.log('mask_accuracy', mask_accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # === SMART DYNAMIC MASKING ===
        old_mask_prob = self.mask_prob_now
        
        # Base progression (your original logic)
        base_mask_prob = self.mask_prob_init + (self.mask_prob_max - self.mask_prob_init) * (self.current_epoch / self.trainer.max_epochs)
        base_mask_prob = min(base_mask_prob, self.mask_prob_max)
        
        # Apply smart adjustments based on recommendations
        mask_prob_adjustment = 0.0
        lr_multiplier = 1.0
        
        for recommendation in recommendations:
            if "BOOST_LEARNING_RATE" in recommendation:
                lr_multiplier = 5.0  # 5x LR boost
                print(f"🚀 Applying LR boost: {lr_multiplier}x")
                
            elif "INCREASE_MASKING" in recommendation:
                mask_prob_adjustment = 0.05  # Add 5% more masking
                print(f"🎯 Increasing masking difficulty: +{mask_prob_adjustment}")
                
            elif "REDUCE_MASKING" in recommendation:
                mask_prob_adjustment = -0.03  # Reduce masking by 3%
                print(f"🔧 Reducing masking difficulty: {mask_prob_adjustment}")
                
            elif "ACCELERATE_MASKING" in recommendation:
                mask_prob_adjustment = 0.03  # Faster progression
                print(f"⚡ Accelerating masking progression: +{mask_prob_adjustment}")
                
            elif "ADD_REGULARIZATION" in recommendation:
                # This is handled in the loss function
                print(f"🛡️ Enabling regularization (label smoothing)")
        
        # Apply masking adjustments
        self.mask_prob_now = base_mask_prob + mask_prob_adjustment
        self.mask_prob_now = max(0.05, min(self.mask_prob_now, 0.4))  # Keep in reasonable bounds
        
        # Apply learning rate adjustments
        if lr_multiplier != 1.0:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * lr_multiplier
                print(f"📈 LR changed: {old_lr:.2e} → {param_group['lr']:.2e}")
        
        # === ADVANCED TRAINING STRATEGIES ===
        
        # Strategy 1: Curriculum Learning - adjust based on performance
        if mask_accuracy > 0.20 and self.current_epoch > 50:
            # High accuracy - can handle more challenge
            curriculum_boost = min(0.05, (mask_accuracy - 0.20) * 0.25)
            self.mask_prob_now += curriculum_boost
            
        elif mask_accuracy < 0.08 and self.current_epoch > 30:
            # Low accuracy - reduce challenge
            curriculum_reduction = max(-0.03, (mask_accuracy - 0.08) * 0.5)
            self.mask_prob_now += curriculum_reduction
        
        # Strategy 2: Adaptive masking based on loss trends
        if hasattr(self.loss_fn, 'loss_history') and len(self.loss_fn.loss_history) > 10:
            recent_losses = self.loss_fn.loss_history[-10:]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            
            if loss_trend > 0.01:  # Loss increasing
                self.mask_prob_now *= 0.95  # Slightly easier
                print(f"📉 Loss increasing, reducing masking: {old_mask_prob:.3f} → {self.mask_prob_now:.3f}")
                
            elif loss_trend < -0.02 and mask_accuracy > 0.12:  # Loss decreasing well
                self.mask_prob_now *= 1.02  # Slightly harder
                print(f"📈 Good loss trend, increasing masking: {old_mask_prob:.3f} → {self.mask_prob_now:.3f}")
        
        # Strategy 3: Stochastic masking variation
        if self.current_epoch % 20 == 0 and self.current_epoch > 0:
            # Every 20 epochs, add some randomness to escape local minima
            noise = torch.randn(1).item() * 0.02  # Small random variation
            self.mask_prob_now += noise
            print(f"🎲 Adding stochastic variation: {noise:+.3f}")
        
        # Ensure masking probability stays in bounds
        self.mask_prob_now = max(0.05, min(self.mask_prob_now, self.mask_prob_max))
        
        # Log the final masking probability
        self.log('mask_prob', self.mask_prob_now, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log masking change if significant
        if abs(self.mask_prob_now - old_mask_prob) > 0.01:
            mask_change = self.mask_prob_now - old_mask_prob
            print(f"🔄 Mask probability adjusted: {old_mask_prob:.3f} → {self.mask_prob_now:.3f} ({mask_change:+.3f})")
        
        # Update the model with new masking probability
        self.model.update_mask_prob(self.mask_prob_now)
        
        # === PERIODIC TRAINING HEALTH CHECKS ===
        
        if self.current_epoch % 25 == 0 and self.current_epoch > 0:
            self.print_training_health_report(mask_accuracy, recommendations)
        
        # === AUTOMATIC CHECKPOINT SUGGESTIONS ===
        
        if mask_accuracy > getattr(self, '_best_accuracy_so_far', 0.0):
            self._best_accuracy_so_far = mask_accuracy
            print(f"🏆 New best accuracy achieved: {mask_accuracy:.4f}")
            
            # Suggest saving checkpoint for very good performance
            if mask_accuracy > 0.25:
                print(f"💾 Consider saving checkpoint - excellent performance!")
        
        super().on_train_epoch_end()
        torch.cuda.empty_cache()

    def print_training_health_report(self, current_accuracy, recommendations):
        """Print a comprehensive training health report"""
        print(f"\n" + "="*60)
        print(f"🏥 TRAINING HEALTH REPORT - Epoch {self.current_epoch}")
        print(f"="*60)
        
        # Performance metrics
        print(f"📊 Performance Metrics:")
        print(f"   Current Accuracy: {current_accuracy:.4f}")
        print(f"   Best Accuracy: {getattr(self, '_best_accuracy_so_far', 0.0):.4f}")
        print(f"   Current Mask Prob: {self.mask_prob_now:.3f}")
        print(f"   Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Training status
        if hasattr(self.loss_fn, 'plateau_detected'):
            status = "🚨 PLATEAU" if self.loss_fn.plateau_detected else "✅ PROGRESSING"
            print(f"   Training Status: {status}")
        
        # Recent trends
        if hasattr(self.loss_fn, 'masked_accuracy_mem_epoch') and len(self.loss_fn.masked_accuracy_mem_epoch) >= 5:
            recent_acc = self.loss_fn.masked_accuracy_mem_epoch[-5:]
            trend = recent_acc[-1] - recent_acc[0]
            trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            print(f"   5-Epoch Trend: {trend_symbol} {trend:+.4f}")
        
        # Active recommendations
        if recommendations:
            print(f"\n🎯 Active Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n✅ No interventions needed - training healthy")
        
        print(f"="*60 + "\n")

        
    def on_epoch_end(self):
        """Epoch end callback"""
        torch.cuda.empty_cache()


    
        
    
    def on_end_metrics_eval(self, mode):
        
        torch.cuda.synchronize()
        return {}
    
    def on_validation_epoch_end(self):
        """Validation epoch end callback"""
        # Calculate and log embedding space separation metrics
        #self._log_embedding_separation()
        pass
    

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler"""
        # Get optimizer settings from hyperparameters
        lr = float(getattr(self.hp, 'lr', 1e-5))  # Lower default learning rate
        weight_decay = float(getattr(self.hp, 'weight_decay', 0.01))
        
        # Create optimizer with better numerical stability
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8  # Increased epsilon for stability
        )
        
        # Store optimizer reference for gradient clipping
        self.optimizer = optimizer
        
        # Calculate total steps for scheduler
        total_steps = self.trainer.estimated_stepping_batches
        
        # Use cosine scheduler with warmup
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 5,  # More conservative max_lr
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=3.0,  # Less aggressive initial decrease
                final_div_factor=10.0,  # Less aggressive final decrease
                anneal_strategy="cos"
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
