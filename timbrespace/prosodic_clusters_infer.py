"""
"""

import torch
import torchaudio
import os


from frames_clustering import ClusteringPipeline, ArrayClass
from prosodic_features.prosodic_features import ProsodicFeatureExtractor



class ProsodicClusterSequences:
    
    def __init__(self, prosodic_cluster_model_dir, max_duration=10,  device="cpu"):
        """
        Initialize the pipeline
        
        Args:
            prosodic_cluster_model_dir: Path to prosodic clustering model directory
            max_duration: Maximum duration of audio signals in seconds
            device: Device to run inference on
        """
        
        self.device = device
        self.max_duration = max_duration
        self.sample_rate = 16000
        self.max_wav_len = max_duration*self.sample_rate
        
        self.prosodic_features_dim = 6  # Prosodic features: f0, f0_width, f1, f2, power, snr
        
        
        self.prosodic_features_weights = {'f0': 1.5, 'f0_width': 1, 'f1': 1.5,  'f2': 0.75, 'power': 2, 'snr': 1}
        
        self.prosodic_features_to_indices = {'f0': 0, 'f0_width': 1, 'f1': 2, 'f2': 3, 'power': 4, 'snr': 5}
        assert (len(self.prosodic_features_weights) == self.prosodic_features_dim), f"Prosodic features weights length mismatch: {len(self.prosodic_features_weights)} vs {self.prosodic_features_dim}"

        
        # Initialize clustering models
        self.clustering_model_prosodic = None
    
    
        self.clustering_model_prosodic = ClusteringPipeline(n_clusters=None, device=self.device)
        if (os.path.exists(prosodic_cluster_model_dir)) and (os.path.isdir(prosodic_cluster_model_dir)) and (os.path.exists(os.path.join(prosodic_cluster_model_dir, "kmeans_model.json"))):
            self.clustering_model_prosodic.load_model(prosodic_cluster_model_dir)
            print("Prosodic Clustering model loaded from", prosodic_cluster_model_dir)
            print("Prosodic n_clusters:", self.clustering_model_prosodic.n_clusters)

        self.prosodic_extractor = ProsodicFeatureExtractor(sample_rate=self.sample_rate, frame_ms=50, hop_ms=16, emphasize_ratio=0.7, output_normalize=True) 
        
        
        dummy_wav = torch.zeros(1, self.max_wav_len, dtype=torch.float32, device='cpu')  # dummy waveform for config
        dummy_features = self.prosodic_extractor.extract(audio_tensor=dummy_wav, input_normalize=False)  # initialize prosodic extractor
        if dummy_features is None or not isinstance(dummy_features, dict):
            print("Prosodic feature extraction failed. Please check the ProsodicFeatureExtractor configuration.")
            print("Ensure the ProsodicFeatureExtractor is properly initialized with correct parameters.")
            print("Exiting...")
            exit()
        self.max_prosodic_frames = max(dummy_features["f0"].shape[0], dummy_features["power"].shape[0], dummy_features["f1"].shape[0], dummy_features["snr"].shape[0])
        print(f"Max prosodic frames: {self.max_prosodic_frames}")


    def resample_waveform(self, audio_signal, original_sample_rate):

        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=self.sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        ) 
        audio_signal = resampler(audio_signal)
        rms = torch.sqrt(torch.mean(audio_signal ** 2)) # rmn normalization
        if rms > 0:
            audio_signal = audio_signal / rms

        return audio_signal

    def _process_prosodic_features(self, audio_batch, audio_lengths=None):
        """
        Process a batch of audio to extract prosodic features
        
        Args:
            audio_batch: Batch of audio waveforms

        Returns:
            prosodic_features: Prosodic features for each audio in the batch
        """

        prosodic_features = torch.zeros((audio_batch.shape[0], self.max_prosodic_frames, self.prosodic_features_dim), dtype=torch.float32, device=self.device)
        # Extract prosodic features
        for b in range(audio_batch.shape[0]):
            audio_in = audio_batch[b].to('cpu')
            if audio_lengths is not None:
                audio_in = audio_in[:audio_lengths[b]]  # trim to length if provided
            _features = self.prosodic_extractor.extract(audio_tensor=audio_in, input_normalize=False)
            if (audio_lengths is None):
                for key in self.prosodic_features_to_indices.keys():
                    assert (_features[key].shape[0] == self.max_prosodic_frames), f"Prosodic features length mismatch for item {b}: {_features[key].shape[0]} vs {self.max_prosodic_frames}"
                    prosodic_features[b, :, self.prosodic_features_to_indices[key]] = torch.from_numpy(_features[key][:self.max_prosodic_frames]).to(self.device) * self.prosodic_features_weights[key]
                
            else:
                for key in self.prosodic_features_to_indices.keys():
                    _prosodic_features_out = torch.from_numpy(_features[key]).to(self.device) * self.prosodic_features_weights[key]
                    assert (_prosodic_features_out.shape[0] <= self.max_prosodic_frames), f"Prosodic features length mismatch for item {b}: {_prosodic_features_out.shape[0]} vs {self.max_prosodic_frames}"
                    prosodic_features[b, :_prosodic_features_out.shape[0], self.prosodic_features_to_indices[key]] = _prosodic_features_out


        return prosodic_features

    def _extract_features_batch(self, audio_batch, audio_lengths):
        """
        Process a batch of audio to extract logits
        
        Args:
            audio_batch: Batch of audio waveforms
            audio_lengths: Lengths of each audio in the batch
            
        Returns:
            prosodic_features: Prosodic features
            output_frames_lens: Lengths of frames for each audio in the batch
        """
        prosodic_features = self._process_prosodic_features(audio_batch, audio_lengths)  # Extract prosodic features
        
        
        output_frames_lens = torch.zeros((audio_lengths.shape[0],), dtype=torch.int32, device=audio_lengths.device)
        for i in range(audio_batch.shape[0]):
            output_frames_lens[i] = self.prosodic_extractor.get_frames_count(audio_lengths[i].item())  # Estimate number of frames for each audio in the batch
            if output_frames_lens[i] > self.max_prosodic_frames:
                output_frames_lens[i] = self.max_prosodic_frames

        return prosodic_features, output_frames_lens
    
    def extract_collated_embeddings(self, dataloader):
        """
        Extract embeddings for training clustering models.
        Returns collated embeddings for both CUPE and prosodic features.
        """
        
        print("Starting embeddings extraction process (collated)...")
        prosodic_features_collation = None
        
        from tqdm import tqdm

        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Extracting phonemes")):
        
            audio_batch, audio_lengths = batch_data
            
                
            # Process audio and get predictions
            prosodic_features, output_frames_lens = self._extract_features_batch(audio_batch, audio_lengths)
            
            # Process each sequence in the batch manually 
            batch_size = output_frames_lens.shape[0]
            
            assert prosodic_features.shape[2] == self.prosodic_features_dim, f"Expected prosodic features_dim {self.prosodic_features_dim}, got {prosodic_features.shape[2]}"
            

            for i in range(batch_size):

                
            
                # Get sequence data
                if output_frames_lens is not None:
                    prosodic_i = prosodic_features[i][:output_frames_lens[i]]
                else:
                    prosodic_i = prosodic_features[i]
                    prosodic_i = prosodic_i.detach()

                if (prosodic_features_collation is None):
                    prosodic_features_collation = ArrayClass(features_dim=self.prosodic_features_dim, init_capacity=int(prosodic_i.shape[0]*10*batch_size), device="cpu")
                prosodic_features_collation.add_batch(prosodic_i.cpu())  # add to features collation
                
                
        # Finalize features collation
        prosodic_features_collation.finalize()
        print(f"Extracted {len(prosodic_features_collation)} prosodic embeddings")
        if len(prosodic_features_collation) == 0:
            raise ValueError("No valid phoneme features were extracted.")

        
        
        return prosodic_features_collation


    def prepare_audio_batch(self, audio_signals, audio_lengths=None, enable_padding=True):
        if audio_signals.dim() == 1:
            audio_batch = audio_signals.unsqueeze(0)
        elif audio_signals.dim() == 2:
            audio_batch = audio_signals
        elif audio_signals.dim() == 3 and audio_signals.shape[1] == 1:
            audio_batch = audio_signals.squeeze(1)
        else:
            raise ValueError(f"Unsupported audio signal dimensions: {audio_signals.dim()}. Expected 1D or 2D tensor.")
        batch_size = audio_batch.shape[0]

        if audio_lengths is not None:
            wav_lens = audio_lengths.clone()  # Clone to avoid modifying the original tensor
            if isinstance(wav_lens, torch.Tensor):
                wav_lens = wav_lens.to(audio_batch.device)  # Ensure wav_lens is on the same device as audio_batch
            elif isinstance(wav_lens, list):
                wav_lens = torch.tensor(wav_lens, dtype=torch.int32, device=audio_batch.device)  # Convert list to tensor
            else:
                raise ValueError(f"Unsupported type for audio_lengths: {type(audio_lengths)}")
            assert wav_lens.shape[0] == batch_size, f"Expected wav_lens shape {batch_size}, got {wav_lens.shape[0]}"
        else:
            wav_lens = torch.zeros((batch_size,), dtype=torch.int32, device=audio_batch.device)  # Initialize wav_lens
            for i in range(batch_size): 
                wav_lens[i] = audio_batch[i].shape[0]  # Set length of each audio in the batch

        if audio_batch.shape[1] > self.max_wav_len:
            raise ValueError(f"Audio batch length {audio_batch.shape[1]} exceeds maximum allowed length {self.max_wav_len}. Please ensure audio signals are within the specified duration of {self.max_duration} seconds.")
        

        if enable_padding:
            if audio_batch.shape[1] < self.max_wav_len:
                # Pad audio_batch to max_wav_len
                padding = torch.zeros((batch_size, self.max_wav_len - audio_batch.shape[1]), dtype=torch.float32, device=audio_batch.device)
                audio_batch = torch.cat((audio_batch, padding), dim=1)
            else:
                # Ensure audio_batch is exactly max_wav_len
                audio_batch = audio_batch[:, :self.max_wav_len]
        # if enable_padding is False, ensure audio_batch is exactly max_wav_len

        assert audio_batch.shape[1] == self.max_wav_len, f"Expected audio batch shape {self.max_wav_len}, got {audio_batch.shape[1]}"

        return audio_batch, wav_lens

    def extract_cluster_sequences(self, audio_signals, audio_lengths=None, return_distances=False, ):
        """
        Extract frame sequences using prosodic clustering models.
        This method processes audio batches and generates prosodic sequences.
        Args:
            audio_signals: 1D, or 2D tensor of audio signals (batch_size, signal_length) must have the sample rate of self.sample_rate (16000)
            audio_lengths: Lengths of each audio in the batch (optional), if None it will be calculated from audio_batch items. Shape: (batch_size,)
            return_distances: If True, returns distances along with sequences.
        Returns:
            None: The sequences are stored in the dataset.
        """

        extracted_sequences = []
        extracted_distances = []

        audio_batch, _audio_lengths = self.prepare_audio_batch(audio_signals, audio_lengths)  # Prepare audio batch
        
        batch_size = audio_batch.shape[0]
        # Process audio and get predictions
        prosodic_features, output_frames_lens = self._extract_features_batch(audio_batch, _audio_lengths)
        
        # Process each sequence in the batch manually 
        assert output_frames_lens.shape[0] == batch_size, f"Expected output_frames_lens shape {batch_size}, got {output_frames_lens.shape[0]}"
        assert prosodic_features.shape[0] == batch_size, f"Expected prosodic_features shape {batch_size}, got {prosodic_features.shape[0]}"
        assert prosodic_features.shape[1] == self.max_prosodic_frames, f"Expected prosodic_features shape {self.max_prosodic_frames}, got {prosodic_features.shape[1]}"
        assert prosodic_features.shape[2] == self.prosodic_features_dim, f"Expected prosodic features_dim {self.prosodic_features_dim}, got {prosodic_features.shape[2]}"

        for i in range(batch_size):
            

            # Get sequence data
            if output_frames_lens is not None:
                prosodic_features_i = prosodic_features[i][:output_frames_lens[i]]
            else:
                prosodic_features_i = prosodic_features[i]
            
            prosodic_features_i = prosodic_features_i.detach()
            

            # Apply dimension reduction if available
            if (self.clustering_model_prosodic.dim_reducer is not None):
                prosodic_features_i = self.clustering_model_prosodic.apply_dimension_reducer(prosodic_features_i.cpu())
            
            # Predict prosodic labels
            labels, distances = self.clustering_model_prosodic.model.predict(prosodic_features_i, return_dists=False)
            
            if return_distances:
                distances = distances.tolist()
                extracted_distances.append(distances.tolist())
            
            # Store sequence
            if labels is not None:
                extracted_sequences.append(labels.tolist())
            else:
                extracted_sequences.append([])  # Append empty sequence if none


        if return_distances:
            return extracted_sequences, extracted_distances
        
        return extracted_sequences





def example_single_clip(path_to_audio="example.wav"):
    """
    Example usage of the ProsodicClusterSequences for a single audio clip.
    """
    
    max_duration = 10  # seconds
    device = "cpu"
    


    prosodic_cluster_model_dir = "models/clustering/libri_train_100_Ju29_n120_uj01d_e62_35h_prosodic"  # Path to prosodic clustering model directory

    # Initialize the pipeline
    extractor = ProsodicClusterSequences(prosodic_cluster_model_dir, max_duration=max_duration, device=device)
    

    audio_signal, sr = torchaudio.load(path_to_audio, normalize=True)  # Load audio signal
    audio_signal = extractor.resample_waveform(audio_signal, original_sample_rate=sr)
    audio_lengths = torch.tensor([audio_signal.shape[1]], dtype=torch.int32, device=device)  # Length of audio signal, shape: [B, 1]

    cluster_sequences = extractor.extract_cluster_sequences(audio_signal, audio_lengths=audio_lengths, return_distances=False)
    print("Input signal length:", len(audio_signal))
    print("Number of label sequences extracted:", len(cluster_sequences))
    print("Length of first label sequence:", len(cluster_sequences[0]))
    print("Extracted sequence:", cluster_sequences[0])


def example():
    """
    Example usage of the ProsodicClusterSequences for a noise audio signal.
    """
    sample_rate = 16000
    max_duration = 10  # seconds
    device = "cpu"
    audio_signal = torch.randn((sample_rate * max_duration,))  # Simulated audio signal





    prosodic_cluster_model_dir = "models/clustering/Aug09/libri_train-clean-100_101hr_Au08_n120_20_35h_prosodic"  # Path to prosodic clustering model directory

    # Initialize the pipeline
    extractor = ProsodicClusterSequences(prosodic_cluster_model_dir, max_duration=max_duration, device=device)

    cluster_sequences = extractor.extract_cluster_sequences(audio_signal, return_distances=False)
    print("Input signal length:", len(audio_signal))
    print("Number of label sequences extracted:", len(cluster_sequences))
    print("Length of first label sequence:", len(cluster_sequences[0]))
    print("Extracted sequence:", cluster_sequences[0])



if __name__ == "__main__":

    torch.manual_seed(42)
    #example()
    example_single_clip(path_to_audio="audio_samples/109867__timkahn__butterfly.wav")
    print("Done!")