# Timbrespace

Speech prosodic feature extraction and embedding generation toolkit for creating interpretable speech style representations.


## What It Does

Timbrespace extracts prosodic features from speech audio and converts them into embeddings that capture speaking style, speaker identity, and emotional characteristics. The system works in three stages:

1. **Feature Extraction**: Extracts 6 prosodic features from audio waveforms
2. **Clustering**: Quantizes features into discrete tokens using K-means clustering
3. **Embedding Generation**: Uses a transformer model to create contextual embeddings from token sequences

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for clustering and model training)
- 16kHz audio input (automatically resampled if different)

## Installation

**Current (Development):**
```bash
# Clone repository
git clone https://github.com/yourusername/timbrespace.git
cd timbrespace

# Install dependencies
pip install torch torchaudio librosa numpy scipy matplotlib scikit-learn numba

# Optional: for UMAP dimensionality reduction
pip install umap-learn
```


## Usage Guide

### Step 1: Extract Prosodic Features

Extract 6 prosodic features from audio files:

```python
from timbrespace.prosodic_features.prosodic_features import ProsodicFeatureExtractor

# Initialize with default settings
extractor = ProsodicFeatureExtractor(
    sample_rate=16000,     # Target sample rate
    frame_ms=50,           # Analysis window size
    hop_ms=16,             # Frame step size
    output_normalize=True  # Normalize features to [0,1]
)

# Load audio file (supports wav, mp3, flac via librosa)
audio_tensor = extractor.load_audio("audio.wav", loader="librosa")

# Extract features for entire audio
features = extractor.extract()

# Extract features for specific time range
features = extractor.extract(start_ms=1000, end_ms=5000)  # 1-5 seconds

# Access individual features
f0 = features['f0']           # Fundamental frequency
f0_width = features['f0_width']  # Voice quality/breathiness
f1 = features['f1']           # First formant
f2 = features['f2']           # Second formant
power = features['power']     # Signal energy
snr = features['snr']         # Signal-to-noise ratio

print(f"Extracted {len(f0)} frames at {len(f0)/extractor.get_frames_count(len(audio_tensor)):.1f} fps")
```

**Feature Details:**
- **f0**: Pitch contour (fundamental frequency)
- **f0_width**: Voice quality measure (bandwidth of F0)
- **f1/f2**: Vowel formant frequencies
- **power**: Voicing strength
- **snr**: Voice clarity measure

### Step 2: Cluster Features into Tokens

Convert continuous features into discrete tokens:

> **Note**: This step requires training a clustering model on your dataset. Pre-trained clustering models will be available on HuggingFace Hub soon.

```python
from timbrespace.frames_clustering import ClusteringPipeline, ArrayClass

# Train clustering model
clustering = ClusteringPipeline(
    n_clusters=120,           # Number of prosodic tokens
    input_features_dim=6,     # 6 prosodic features
    device="cuda"             # Use GPU for speed
)

# Collect features from multiple audio files
feature_collection = ArrayClass(features_dim=6, init_capacity=10000)

# Add features from multiple files
for audio_file in audio_files:
    features = extractor.extract(audio_tensor=load_audio(audio_file))
    # Stack features into 6D vectors
    feature_vectors = np.stack([
        features['f0'], features['f0_width'], features['f1'],
        features['f2'], features['power'], features['snr']
    ], axis=1)
    feature_collection.add_batch(torch.from_numpy(feature_vectors))

feature_collection.finalize()

# Train clustering model
clustering.train(
    features_collation=feature_collection,
    clustering_model_dir="models/clustering",
    reduced_dims=None,  # Set to int for dimensionality reduction
    max_iter=1000
)
```

### Step 3: Generate Token Sequences

Convert audio to prosodic token sequences:

```python
from timbrespace.prosodic_clusters_infer import ProsodicClusterSequences

# Load trained clustering model
cluster_extractor = ProsodicClusterSequences(
    prosodic_cluster_model_dir="models/clustering",
    max_duration=30,  # Maximum audio length in seconds
    device="cpu"
)

# Process single audio file
audio_signal, sr = torchaudio.load("audio.wav")
audio_signal = cluster_extractor.resample_waveform(audio_signal, sr)

# Extract prosodic token sequence
sequences = cluster_extractor.extract_cluster_sequences(
    audio_signal,
    return_distances=False
)

print(f"Token sequence: {sequences[0]}")
# Output: [45, 67, 23, 89, 12, ...]  # Sequence of prosodic tokens
```

### Step 4: Train/Use Transformer Model

Generate contextual embeddings from token sequences:

> **Note**: Pre-trained models for `SequenceEncoderPipeline` which are available on [Google Drive](https://drive.google.com/drive/folders/1Y3qfH3qVpqIRJVGNhy2_ZwYBox3mF13S?usp=sharing). It's a lightining module that wraps around `ProsodicSequenceEncoder`.

```python
from timbrespace.p3vs2prbert.model_timbrespace import ProsodicSequenceEncoder

# Initialize transformer model
model = ProsodicSequenceEncoder(
    vocab_size=122,        # 120 clusters + padding + mask tokens
    hidden_dim=768,        # Embedding dimension
    num_layers=8,          # Transformer layers
    nhead=8,               # Attention heads
    pos_max_len=2000,      # Maximum sequence length
    padding_token_id=-100  # Padding token ID
)

# Convert sequences to tensor
sequences_tensor = torch.tensor(sequences, dtype=torch.long)

# Training mode (with masking for self-supervised learning)
model.train()
logits, hidden_states, target_mask, original_tokens = model(
    sequences_tensor,
    training=True
)

# Inference mode (generate embeddings)
model.eval()
with torch.no_grad():
    embeddings = model(sequences_tensor, training=False)
    # embeddings shape: [batch_size, hidden_dim]

print(f"Generated embeddings shape: {embeddings.shape}")
```
> **Work in Progress**
>
> This repository is currently under active development. We are preparing:
> - **Pre-trained models** on HuggingFace Hub
> - **Complete documentation** and tutorials
>
> The code is functional but the API may change before the official release.


## Module Reference

### ProsodicFeatureExtractor (`prosodic_features/prosodic_features.py`)

Main class for extracting prosodic features from audio.

**Key Methods:**
- `load_audio(path, loader="librosa")` - Load audio file
- `extract(start_ms=0, end_ms=-1)` - Extract features from audio segment
- `get_frames_count(signal_length)` - Calculate expected frame count
- `plot_features(features, save_path)` - Visualize extracted features

**Parameters:**
- `sample_rate=16000` - Target sample rate
- `frame_ms=50` - Analysis window size in milliseconds
- `hop_ms=16` - Frame step size in milliseconds
- `emphasize_ratio=0.7` - Pre-emphasis filter coefficient
- `output_normalize=True` - Normalize features to [0,1] range

### ClusteringPipeline (`frames_clustering.py`)

K-means clustering for prosodic feature quantization.

**Key Methods:**
- `train(features, model_dir, max_iter=1000)` - Train clustering model
- `load_model(model_dir)` - Load pre-trained clustering model
- `clustering_quality_analysis(features)` - Evaluate clustering quality

**Parameters:**
- `n_clusters=120` - Number of prosodic tokens
- `input_features_dim=6` - Feature vector dimension
- `device="cuda"` - Computing device

### ProsodicClusterSequences (`prosodic_clusters_infer.py`)

Convert audio to prosodic token sequences using trained models.

**Key Methods:**
- `extract_cluster_sequences(audio, lengths=None)` - Generate token sequences
- `prepare_audio_batch(audio, lengths=None)` - Prepare audio for processing

**Parameters:**
- `prosodic_cluster_model_dir` - Path to trained clustering model
- `max_duration=10` - Maximum audio length in seconds
- `device="cpu"` - Computing device

### ProsodicSequenceEncoder (`p3vs2prbert/model_timbrespace.py`)

Transformer model for prosodic sequence modeling.

**Key Methods:**
- `forward(x, lengths=None, training=False)` - Process token sequences
- `update_mask_prob(prob)` - Adjust masking probability during training

**Parameters:**
- `vocab_size=122` - Vocabulary size (clusters + special tokens)
- `hidden_dim=768` - Embedding/hidden dimension
- `num_layers=8` - Number of transformer layers
- `nhead=8` - Number of attention heads
- `pos_max_len=2000` - Maximum sequence length

## Configuration Options

### Feature Extraction Settings

```python
# High-quality settings (slower)
extractor = ProsodicFeatureExtractor(
    frame_ms=50,
    hop_ms=10,      # Higher temporal resolution
    smoothing_level='heavy'
)

# Fast settings (lower quality)
extractor = ProsodicFeatureExtractor(
    frame_ms=40,
    hop_ms=20,      # Lower temporal resolution
    smoothing_level='light'
)
```

### Clustering Configuration

```python
# Large vocabulary (more detailed prosodic patterns)
clustering = ClusteringPipeline(n_clusters=256)

# Small vocabulary (coarser patterns, faster training)
clustering = ClusteringPipeline(n_clusters=64)

# With dimensionality reduction
clustering.train(
    features_collation=features,
    clustering_model_dir="models",
    reduced_dims=50,           # Reduce from 6D to 50D
    dim_reducer_method='pca'   # or 'umap'
)
```

### Model Architecture

```python
# Large model (better quality)
model = ProsodicSequenceEncoder(
    hidden_dim=1024,
    num_layers=12,
    nhead=16
)

# Small model (faster inference)
model = ProsodicSequenceEncoder(
    hidden_dim=512,
    num_layers=6,
    nhead=8
)
```

## File Formats

**Input Audio:**
- Supported: WAV, MP3, FLAC (via librosa)
- Recommended: 16kHz mono WAV files
- Auto-resampling: Yes

**Model Files:**
- Clustering model: `kmeans_model.json` + `cluster_centers.pt`
- Transformer model: Standard PyTorch `.pt` files
- Reduction models: Pickle `.pkl` files

**Output Formats:**
- Features: Dict of NumPy arrays
- Token sequences: Lists of integers
- Embeddings: PyTorch tensors

## Performance Notes

- **GPU Acceleration**: Clustering and model training benefit significantly from GPU
- **Memory Usage**: ~1GB per 100k feature vectors during clustering
- **Processing Speed**: ~50x real-time for feature extraction on CPU
- **Model Size**: Transformer model ~200MB (52M parameters), clustering model ~50KB
