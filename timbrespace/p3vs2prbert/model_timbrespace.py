import torch
from torch import nn
import torch.nn.functional as F
import math
import random


import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


class PositionalEncoding(nn.Module):
    """Positional encoding with support for masking and long sequences"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Boolean mask for padding [batch_size, seq_len]
        """
        x = x + self.pe[:, :x.size(1)]
        
        if mask is not None:
            # Apply mask - set masked positions to zero
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            x = x * mask_expanded
            
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """Embedding layer for discrete K-means labels with proper validation"""
    def __init__(self, vocab_size, hidden_dim, padding_token_id=-100):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.padding_token_id = padding_token_id
        
        # Create embedding with padding index
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.padding_token_id)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # FIXED: Proper validation logic
        # Valid tokens are: [0, vocab_size-1] OR padding_token_id
        valid_range_mask = (x >= 0) & (x < self.vocab_size)
        padding_mask = (x == self.padding_token_id)
        valid_mask = valid_range_mask | padding_mask
        
        if not valid_mask.all():
            invalid_mask = ~valid_mask
            invalid_tokens = x[invalid_mask]
            print(f"Invalid token IDs detected:")
            print(f"  Invalid values: {invalid_tokens.unique()}")
            print(f"  Min value in batch: {x.min().item()}")
            print(f"  Max value in batch: {x.max().item()}")
            print(f"  Expected range: [0, {self.vocab_size-1}] or padding_token_id={self.padding_token_id}")
            print(f"  Vocab size: {self.vocab_size}")
            
            # Show positions of invalid tokens
            invalid_positions = invalid_mask.nonzero(as_tuple=True)
            print(f"  Invalid positions (first 10): {[(invalid_positions[0][i].item(), invalid_positions[1][i].item()) for i in range(min(10, len(invalid_positions[0])))]}")
            
            # For debugging - show what tokens are causing issues
            unique_invalid = invalid_tokens.unique()
            for token in unique_invalid:
                count = (invalid_tokens == token).sum().item()
                print(f"    Token {token}: appears {count} times")
        
        # Handle padding tokens by mapping them to index 0 temporarily
        # This prevents embedding lookup errors
        x_mapped = torch.where(x == self.padding_token_id, 0, x)
        
        # FIXED: Clamp to valid range more carefully
        x_mapped = torch.clamp(x_mapped, 0, self.vocab_size - 1)
        
        # Scale embeddings by sqrt(hidden_dim)
        embedded = self.embedding(x_mapped) * math.sqrt(self.hidden_dim)
        
        # Zero out embeddings for padding positions
        padding_mask = (x == self.padding_token_id).unsqueeze(-1).float()
        embedded = embedded * (1 - padding_mask)
        
        normalized = self.layer_norm(embedded)
        return normalized


class MaskingStrategy(nn.Module):
    """Applies different masking strategies for unsupervised learning with discrete tokens"""
    def __init__(self, mask_prob=0.15, mask_span_dist={10: 0.5, 5: 0.25, 3: 0.15, 1: 0.1}, 
                 vocab_size=256, mask_token_id=257, padding_token_id=256):
        super(MaskingStrategy, self).__init__()
        self.mask_prob = mask_prob
        self.mask_span_dist = mask_span_dist
        self.base_vocab_size = vocab_size  # Only base vocabulary for random sampling
        self.mask_token_id = mask_token_id
        self.padding_token_id = padding_token_id
        
    def forward(self, x, training=True):
        """
        Args:
            x: Input tensor of token ids [batch_size, seq_len]
            training: Whether in training mode
            
        Returns:
            masked_x: Masked input tensor with mask tokens
            mask: Boolean mask indicating non-masked positions [batch_size, seq_len]
            target_mask: Boolean mask for computing loss [batch_size, seq_len]
        """
        if not training:
            # During inference, no masking is applied
            batch_size, seq_len = x.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            return x, mask, target_mask
        
        batch_size, seq_len = x.shape
        
        # Create a copy for masking
        masked_x = x.clone()
        
        # Initialize mask tensors
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Apply masking for each sequence in the batch
        for i in range(batch_size):
            # Skip padding tokens
            valid_positions = (x[i] != self.padding_token_id).nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
                
            valid_seq_len = len(valid_positions)
            
            # Calculate number of tokens to mask
            num_masked = max(1, int(valid_seq_len * self.mask_prob))
            masked_indices = set()
            
            # Apply span masking
            attempts = 0
            while len(masked_indices) < num_masked and attempts < 100:
                attempts += 1
                
                # Choose span length based on distribution
                span_candidates = list(self.mask_span_dist.keys())
                span_probabilities = list(self.mask_span_dist.values())
                span_length = random.choices(span_candidates, weights=span_probabilities, k=1)[0]
                
                # Choose starting position within valid positions
                if valid_seq_len < span_length:
                    span_length = valid_seq_len
                    
                start_pos_idx = random.randint(0, max(0, valid_seq_len - span_length))
                
                # Add indices to masked set
                for j in range(span_length):
                    if start_pos_idx + j < valid_seq_len and len(masked_indices) < num_masked:
                        actual_idx = valid_positions[start_pos_idx + j].item()
                        masked_indices.add(actual_idx)
            
            # Apply masking strategy: 80% [MASK], 10% random, 10% unchanged
            for idx in masked_indices:
                target_mask[i, idx] = True
                mask[i, idx] = False
                
                rand_val = random.random()
                if rand_val < 0.8:
                    masked_x[i, idx] = self.mask_token_id
                elif rand_val < 0.9:
                    # FIXED: Only sample from base vocabulary
                    masked_x[i, idx] = random.randint(0, self.base_vocab_size - 1)
                # else: keep original token
            
        
        return masked_x, mask, target_mask

class ProsodicSequenceEncoder(nn.Module):
    """
    Sequence encoder for discrete phoneme tokens with mask-based unsupervised learning
    """
    def __init__(self, pos_max_len=2000, hidden_dim=512, 
                 num_layers=6, nhead=8, dropout_rate=0.1, vocab_size=256,
                 mask_prob=0.15, mask_span_dist={10: 0.5, 5: 0.25, 3: 0.15, 1: 0.1},
                 padding_token_id=-100):
        super(ProsodicSequenceEncoder, self).__init__()
        
        # Define token IDs clearly upfront
        # FIXED: Clear token ID management
        self.base_vocab_size = vocab_size  # e.g., 122 (includes your phoneme tokens)
        self.padding_token_input = padding_token_id  # -100 from input
        
        # For your case: if vocab_size=122, then:
        # - Tokens 0-121 are valid phoneme tokens
        # - Token 122 will be padding (internal)
        # - Token 123 will be mask token
        self.padding_token_id = vocab_size      # 122
        self.mask_token_id = vocab_size + 1     # 123
        
        # Total vocabulary for embedding layer
        self.total_vocab_size = vocab_size + 2  # 124
        
        self.hidden_dim = hidden_dim
        self.pos_max_len = pos_max_len

        # Token embedding with total vocabulary
        self.token_embedding = TokenEmbedding(
            self.total_vocab_size,  # 124 - handles tokens 0-123
            hidden_dim, 
            padding_token_id=self.padding_token_id  # 122
        )
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout_rate, max_len=self.pos_max_len)
        
        # Masking strategy
        self.masking = MaskingStrategy(
            mask_prob=mask_prob, 
            mask_span_dist=mask_span_dist,
            vocab_size=self.base_vocab_size,  # 122 - only sample from real tokens
            mask_token_id=self.mask_token_id,  # 123
            padding_token_id=self.padding_token_id  # 122
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout_rate, 
            batch_first=True,
            activation='gelu'  # Using GELU for better performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
        # Prediction head outputs only base vocabulary (no special tokens)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, self.base_vocab_size)  # Only predict base tokens
        )
        
        # Layer normalization for final output
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def create_padding_mask(self, x, lengths=None):
        """Create a mask for padding positions"""
        if lengths is not None:
            batch_size, seq_len = x.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            
            for i, length in enumerate(lengths):
                if length < seq_len:
                    mask[i, length:] = False
        else:
            # Create mask based on padding token
            mask = (x != self.padding_token_id)
            
        return mask

    def update_mask_prob(self, new_mask_prob):
        """Update the mask probability dynamically"""
        self.masking.mask_prob = new_mask_prob
        #print(f"Updated mask probability to: {new_mask_prob}")
    
    def forward(self, x, lengths=None, training=False):
        """
        Forward pass with masking for pretraining
        
        Args:
            x: Input token sequences [batch_size, seq_len] of token IDs
            lengths: Optional sequence lengths for masking
            training: Whether in training mode
            
        Returns:
            If training:
                logits: Predicted token logits [batch_size, seq_len, vocab_size]
                transformer_output: Contextual representations [batch_size, seq_len, hidden_dim]
                target_mask: Mask indicating positions to compute loss on
                original_tokens: Original tokens for computing loss
            If not training:
                pooled_output: Sequence representation [batch_size, hidden_dim]
        """
        batch_size, seq_len = x.shape
        

        x = torch.where(x == self.padding_token_input, self.padding_token_id, x)
        
        # FIXED: Proper validation - tokens should be in [0, base_vocab_size-1] or padding_token_id
        valid_base_tokens = (x >= 0) & (x < self.base_vocab_size)  # 0-121
        valid_padding = (x == self.padding_token_id)  # 122
        valid_mask = valid_base_tokens | valid_padding
        
        if not valid_mask.all():
            invalid_tokens = x[~valid_mask].unique()
            print(f"❌ Invalid tokens in model input: {invalid_tokens}")
            print(f"   Expected: [0, {self.base_vocab_size-1}] or {self.padding_token_id}")
            print(f"   Got range: [{x.min().item()}, {x.max().item()}]")
            
            # Clamp invalid tokens to valid range
            x = torch.clamp(x, 0, self.base_vocab_size - 1)
            print(f"   Clamped to: [{x.min().item()}, {x.max().item()}]")
        
        # Store original tokens for loss computation
        original_tokens = x.clone()
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x, lengths)
        
        # Apply masking strategy (only in training)
        if training:
            masked_tokens, mask, target_mask = self.masking(x, training)
            
            # Validate masked tokens
            valid_masked = (masked_tokens >= 0) & (masked_tokens < self.total_vocab_size)
            if not valid_masked.all():
                print(f"❌ Invalid masked tokens: {masked_tokens[~valid_masked].unique()}")
                masked_tokens = torch.clamp(masked_tokens, 0, self.total_vocab_size - 1)
        else:
            masked_tokens = x
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            
        # Convert tokens to embeddings
        embedded = self.token_embedding(masked_tokens)
        
        # Apply positional encoding
        pos_encoded = self.pos_encoder(embedded, padding_mask if training else None)
        
        # Create transformer attention mask from padding mask
        attn_mask = None
        if padding_mask is not None:
            attn_mask = ~padding_mask  # Invert for transformer's expected format
            
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attn_mask)
        transformer_output = self.final_norm(transformer_output)
        
        if training:
            # Predict original tokens
            logits = self.prediction_head(transformer_output)
            return logits, transformer_output, target_mask, original_tokens
        else:
            # For inference, return a pooled representation
            # Apply mean pooling over the sequence (accounting for padding)
            if padding_mask is not None:
                # Multiply by mask and divide by sum
                expanded_mask = padding_mask.unsqueeze(-1).float()
                masked_output = transformer_output * expanded_mask
                pooled_output = masked_output.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).float()
            else:
                # Simple mean pooling if no mask
                pooled_output = transformer_output.mean(dim=1)
                
            return pooled_output