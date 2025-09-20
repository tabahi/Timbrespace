import torch
import torch.nn as nn
import torch.nn.functional as F
# Let's call it "prosodic token masking loss"
class MaskedLanguageModelLoss(nn.Module):
    """
    Loss function with smart training adaptations
    """
    def __init__(self, blank_token_id, vocab_size, padding_token_id=-100,
                 blank_penalty_weight=0.5):
        super(MaskedLanguageModelLoss, self).__init__()
        
        self.blank_token_id = blank_token_id
        self.vocab_size = vocab_size
        self.padding_token_id = padding_token_id
        self.blank_penalty_weight = blank_penalty_weight
        
        # Standard cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=padding_token_id, reduction='none')

        # Tracking metrics
        self.masked_accuracy_mem = []
        self.masked_accuracy_mem_epoch = []
        self.loss_history = []
        self.blank_ratio_history = []
        
        # Adaptive training parameters
        self.plateau_patience = 5  # Epochs to wait before detecting plateau
        self.plateau_threshold = 0.005  # Minimum improvement to not be plateau
        self.last_improvement_epoch = 0
        self.best_accuracy = 0.0
        
        # Learning signals
        self.needs_lr_boost = False
        self.needs_harder_masking = False
        self.needs_regularization = False
        self.plateau_detected = False

    def detect_plateau(self, current_epoch):
        """Detect if training has plateaued and suggest interventions"""
        if len(self.masked_accuracy_mem_epoch) < self.plateau_patience:
            return False
            
        recent_accuracies = self.masked_accuracy_mem_epoch[-self.plateau_patience:]
        max_recent = max(recent_accuracies)
        min_recent = min(recent_accuracies)
        improvement = max_recent - min_recent
        
        # Check if we've improved significantly
        current_accuracy = recent_accuracies[-1]
        if current_accuracy > self.best_accuracy + self.plateau_threshold:
            self.best_accuracy = current_accuracy
            self.last_improvement_epoch = current_epoch
            self.plateau_detected = False
            return False
        
        # Check for plateau
        epochs_without_improvement = current_epoch - self.last_improvement_epoch
        if epochs_without_improvement >= self.plateau_patience and improvement < self.plateau_threshold:
            self.plateau_detected = True
            
            # Determine what intervention is needed
            if current_accuracy < 0.15:  # Very low accuracy
                self.needs_lr_boost = True
            elif current_accuracy < 0.25:  # Moderate accuracy
                self.needs_harder_masking = True
            else:  # Higher accuracy, need regularization
                self.needs_regularization = True
                
            return True
        
        return False

    def get_adaptive_blank_penalty(self, current_accuracy):
        """Adjust blank penalty based on training progress"""
        if self.blank_penalty_weight == 0:
            return 0.0  # Penalty disabled
            
        base_penalty = self.blank_penalty_weight
        
        # Reduce penalty if accuracy is very low (model struggling)
        if current_accuracy < 0.10:
            return base_penalty * 0.5
        # Increase penalty if accuracy is good but plateau detected
        elif self.plateau_detected and current_accuracy > 0.15:
            return base_penalty * 1.5
        
        return base_penalty

    def compute_label_smoothing_loss(self, logits, targets, target_lens, smoothing=0.1):
        """Add label smoothing when regularization is needed"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create valid mask
        valid_mask = self.create_valid_mask(targets, target_lens)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        # Create smoothed targets
        confidence = 1.0 - smoothing
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / vocab_size)
        true_dist.scatter_(2, targets.unsqueeze(2), confidence)
        
        # Apply to valid positions only
        log_probs = F.log_softmax(logits, dim=-1)
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        
        smoothing_loss = -(true_dist * log_probs * valid_mask_expanded).sum() / valid_mask.sum()
        return smoothing_loss

    def on_train_epoch_end(self, epoch, max_epochs):
        """epoch end callback  with smart adaptations"""

        # Store the average masked accuracy for the epoch
        avg_masked_accuracy = 0.0
        if self.masked_accuracy_mem:
            avg_masked_accuracy = sum(self.masked_accuracy_mem) / len(self.masked_accuracy_mem)
            self.masked_accuracy_mem_epoch.append(avg_masked_accuracy)
            
        # Reset memory for masked accuracy at the end of each epoch
        self.masked_accuracy_mem = []
        
        # Keep only recent history for memory efficiency
        if len(self.masked_accuracy_mem_epoch) > 20:
            self.masked_accuracy_mem_epoch = self.masked_accuracy_mem_epoch[-20:]
            
        # Detect plateau and plan interventions
        plateau_detected = self.detect_plateau(epoch)
        
        # Generate training recommendations
        recommendations = self.generate_training_recommendations(epoch, avg_masked_accuracy)
        
        # Print detailed status
        print(f"\n=== Epoch {epoch+1}/{max_epochs} Training Analysis ===")
        print(f"Average Masked Accuracy: {avg_masked_accuracy:.4f}")
        print(f"Best Accuracy So Far: {self.best_accuracy:.4f}")
        print(f"Epochs Since Last Improvement: {epoch - self.last_improvement_epoch}")
        
        if plateau_detected:
            print(f"🚨 PLATEAU DETECTED! Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        else:
            print("✅ Training progressing normally")
            
        if recommendations:
            print(f"📋 Current Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        return avg_masked_accuracy, recommendations

    def generate_training_recommendations(self, epoch, current_accuracy):
        """Generate specific training recommendations based on current state"""
        recommendations = []
        
        if self.needs_lr_boost:
            recommendations.append("BOOST_LEARNING_RATE: Increase LR by 3-5x")
            self.needs_lr_boost = False  # Reset flag
            
        if self.needs_harder_masking:
            recommendations.append("INCREASE_MASKING: Raise mask probability")
            self.needs_harder_masking = False  # Reset flag
            
        if self.needs_regularization:
            recommendations.append("ADD_REGULARIZATION: Enable label smoothing")
            self.needs_regularization = False  # Reset flag
            
        # Additional recommendations based on training stage
        if epoch > 50 and current_accuracy < 0.08:
            recommendations.append("ARCHITECTURE_CHECK: Consider model capacity issues")
            
        if epoch > 100 and current_accuracy < 0.15:
            recommendations.append("DATA_CHECK: Verify data quality and labels")
            
        # Dynamic masking recommendations
        if len(self.masked_accuracy_mem_epoch) >= 3:
            recent_trend = self.masked_accuracy_mem_epoch[-1] - self.masked_accuracy_mem_epoch[-3]
            if recent_trend < -0.01:  # Decreasing accuracy
                recommendations.append("REDUCE_MASKING: Accuracy declining, ease masking")
            elif recent_trend > 0.02 and current_accuracy > 0.12:  # Strong improvement
                recommendations.append("ACCELERATE_MASKING: Good progress, increase difficulty faster")
        
        return recommendations
        
    def create_valid_mask(self, tokens, target_lens):
        """Create mask for valid (non-padding) positions using target_lens"""
        batch_size, seq_len = tokens.shape
        valid_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        for i, length in enumerate(target_lens):
            if length > 0:
                valid_mask[i, :length] = True
                
        return valid_mask
    
    def forward(self, model_output, training=True):
        """Enhanced forward pass with adaptive components"""
        logits, transformer_output, target_mask, original_tokens, target_lens = model_output
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # === 1. CREATE VALID POSITION MASK ===
        valid_mask = self.create_valid_mask(original_tokens, target_lens)
        total_valid_positions = valid_mask.sum().float()
        
        if total_valid_positions == 0:
            return torch.tensor(0.0, device=logits.device), {}, {}
        
        # === 2. RECONSTRUCTION LOSS ===
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = original_tokens.view(-1)
        
        token_losses = self.ce_loss(logits_flat, targets_flat).view(batch_size, seq_len)
        valid_losses = token_losses * valid_mask.float()
        reconstruction_loss = valid_losses.sum() / total_valid_positions
        
        # === 3. BLANK PENALTY (CONDITIONAL) ===
        predictions = torch.argmax(logits, dim=-1)
        blank_predictions = (predictions == self.blank_token_id) & valid_mask
        blank_targets = (original_tokens == self.blank_token_id) & valid_mask
        pred_blank_ratio = blank_predictions.sum().float() / total_valid_positions
        target_blank_ratio = blank_targets.sum().float() / total_valid_positions
        
        # Get current accuracy for adaptive penalty
        masked_positions = target_mask.sum().float()
        current_masked_accuracy = 0.0
        if masked_positions > 0:
            correct_masked = ((predictions == original_tokens) & target_mask).sum().float()
            current_masked_accuracy = correct_masked / masked_positions


        blank_deviation = pred_blank_ratio / (target_blank_ratio + 1e-8)
        # Apply blank penalty only if weight > 0
        if self.blank_penalty_weight > 0:
            adaptive_penalty_weight = self.get_adaptive_blank_penalty(current_masked_accuracy.item())

            
            blank_penalty = adaptive_penalty_weight * blank_deviation
        else:
            blank_penalty = torch.tensor(0.0, device=logits.device)
        
        # === 4. TOTAL LOSS ===
        total_loss = reconstruction_loss + blank_penalty
        
        # Add label smoothing if regularization is needed
        if self.needs_regularization and training:
            smoothing_loss = self.compute_label_smoothing_loss(logits, original_tokens, target_lens)
            total_loss = total_loss + 0.1 * smoothing_loss
        
        # === 5. STATISTICS ===
        token_counts = torch.bincount(predictions[valid_mask], minlength=vocab_size)
        vocab_usage = (token_counts > 0).sum().float() / vocab_size

        if training:
            self.masked_accuracy_mem.append(current_masked_accuracy.item())
            
        # Store current metrics for plateau detection
        if training:
            self.loss_history.append(total_loss.item())
            self.blank_ratio_history.append(pred_blank_ratio.item())
            
        loss_components = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(), 
            'blank_penalty': blank_penalty.item(),
            'pred_blank_ratio': pred_blank_ratio.item(),
            'target_blank_ratio': target_blank_ratio.item(),
            'blank_deviation': blank_deviation.item(),
            'masked_accuracy': current_masked_accuracy.item() if isinstance(current_masked_accuracy, torch.Tensor) else current_masked_accuracy,
            'vocab_usage_ratio': vocab_usage.item(),
            'total_valid_positions': int(total_valid_positions.item()),
            'total_masked_positions': int(masked_positions.item()) if isinstance(masked_positions, torch.Tensor) else int(masked_positions),
            'plateau_detected': self.plateau_detected,
        }
        
        debug_info = {
            #'target_lens': target_lens.cpu().tolist(),
            'blank_token_predictions': blank_predictions.sum().item(),
            #'sequence_lengths': [int(length) for length in target_lens],
            'avg_sequence_length': target_lens.float().mean().item(),
            
        }
        
        return total_loss, loss_components, debug_info