"""
Clustering Pipeline with GPU-Accelerated K-means

This clustering model takes prosodic features and clusters them using K-means.
The model is designed to run on GPU for faster processing and includes feature importance analysis.
The pipeline includes:
1. Training phase: Load phoneme features, train K-means model, and save the model.
2. Inference phase: Load the trained model and predict cluster labels for new phoneme features.
3. Feature importance analysis: Analyze and visualize the importance of each feature in the clustering process.

"""

import torch
import os
import numpy as np
import json
from sklearn.decomposition import PCA
import pickle
from gpu_kmeans_nr import GPUKMeans





class ArrayClass:
    
    def __init__(self, features_dim=None, init_capacity=1000, device='cpu', fill_value=None, reloaded_features=None):
        
        self.capacity = init_capacity
        self.increase_by = init_capacity
        self._fill_value = fill_value

        if reloaded_features is not None:
            if isinstance(reloaded_features, np.ndarray):
                reloaded_features = torch.tensor(reloaded_features, dtype=torch.float, device=self.device)
            
            self.features = reloaded_features
            self.device = reloaded_features.device
            self.features_dim = reloaded_features.shape[1]
            self.idx = reloaded_features.shape[0]
            return
        elif features_dim is None:
            raise ValueError("Either features_dim or reloaded_features must be provided")
        
        self.features_dim = features_dim
        self.device = device
        self.features = torch.zeros((self.capacity, self.features_dim), dtype=torch.float, device=self.device) #can use either torch or np
        
        if fill_value is not None:
            self.features.fill_(fill_value)
            
        self.idx = 0

    def increase_capacity(self):
        new_capacity = int(self.capacity + self.increase_by)
        features_new = torch.zeros((new_capacity, self.features_dim), dtype=torch.float, device=self.device)

        if self._fill_value is not None:
            features_new.fill_(self._fill_value)

        features_new[:self.capacity] = self.features
        self.features = features_new
        del features_new

        self.capacity = new_capacity

    def add(self, features):
        idx = self.idx
        self.features[idx,:] = features
        self.idx += 1

        if (self.idx == self.capacity): self.increase_capacity()


    def add_batch(self, features):
        batch_size = features.shape[0]
        if (self.idx + batch_size >= self.capacity): self.increase_capacity()

        self.features[self.idx:self.idx+batch_size,:] = features
        self.idx += batch_size
        if (self.idx > self.capacity):
            raise Exception("Error: ArrayClass capacity exceeded")
        
        if (self.idx + batch_size >= self.capacity): self.increase_capacity()

    def __len__(self):
        return self.idx
    
    def finalize(self):
        self.features = self.features[:self.idx]
        self.capacity = self.idx


class ClusteringPipeline:
    """
    Integrated class for phoneme clustering that handles both training and inference
    """
    def __init__(self, n_clusters=50, input_features_dim=90, dtype=torch.float32, device="cuda:0"):
        """
        Initialize the phoneme clustering pipeline
        
        Args:
            n_clusters: Number of clusters for K-means
            device: Device to run training/inference on
        """
        self.n_clusters = n_clusters
        self.input_features_dim = input_features_dim
        self.device = device
        self.dtype = dtype
        self.model = GPUKMeans(n_clusters=n_clusters, device=device, dtype=self.dtype)
        
        self.feature_names = None
    

    
        self.reduced_dims=None  # reduced_dims: Optional number of dimensions to reduce the feature space to before clustering (0=auto, None=do not reduce)
        self.dim_reducer = None # save trained reducer model if used
        self.reducer_path = 'reducer.pkl'
        self.model_dir = "./cluster_model"


    def auto_dims_eigen_plot(self, X, plot_path, wandblogger=None):
        """
        Generate eigenvalue plot for PCA to determine optimal number of dimensions.
        Save the plot and return suggested number of dimensions.
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            plot_path: Path to save the eigenvalue plot
            wandblogger: Optional wandb logger object
            
        Returns:
            n_dims: Suggested number of dimensions based on elbow method and variance explained
        """
        print("Analyzing eigenvalues to determine optimal dimensionality...")
        
        import matplotlib.pyplot as plt

        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Ensure we don't try to compute more components than samples or features
        max_components = min(X_np.shape[0], X_np.shape[1], 200)  # Limit to 200 for efficiency
        
        # Fit PCA to get eigenvalues
        pca_temp = PCA(n_components=max_components)
        pca_temp.fit(X_np)
        
        # Get explained variance ratio and cumulative variance
        explained_variance_ratio = pca_temp.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Eigenvalues (explained variance)
        ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', markersize=4)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot (Eigenvalues)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, min(50, len(explained_variance_ratio)))  # Show first 50 components
        
        # Plot 2: Cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=4)
        ax2.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% variance')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
        ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, label='98% variance')
        ax2.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, min(100, len(cumulative_variance)))
        
        # Plot 3: Elbow detection (second derivative)
        if len(explained_variance_ratio) > 2:
            # Calculate second derivative for elbow detection
            first_deriv = np.diff(explained_variance_ratio)
            second_deriv = np.diff(first_deriv)
            
            ax3.plot(range(2, len(second_deriv) + 2), second_deriv, 'go-', markersize=4)
            ax3.set_xlabel('Principal Component')
            ax3.set_ylabel('Second Derivative')
            ax3.set_title('Elbow Detection (Second Derivative)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(2, min(50, len(second_deriv) + 2))
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log plot to wandb if logger provided
        if wandblogger is not None:
            try:
                import wandb
                wandblogger.log({"dimensionality_analysis_plot": wandb.Image(fig)})
                print("✓ Dimensionality analysis plot logged to wandb")
            except Exception as e:
                print(f"Warning: Could not log plot to wandb: {e}")
        
        plt.close()
        print(f"Eigenvalue plot saved to: {plot_path}")
        
        # Determine optimal number of dimensions using multiple criteria
        n_dims_suggestions = {}
        
        # Criterion 1: 90% variance explained
        idx_90 = np.where(cumulative_variance >= 0.90)[0]
        if len(idx_90) > 0:
            n_dims_suggestions['90%_variance'] = idx_90[0] + 1
        
        # Criterion 2: 95% variance explained
        idx_95 = np.where(cumulative_variance >= 0.95)[0]
        if len(idx_95) > 0:
            n_dims_suggestions['95%_variance'] = idx_95[0] + 1
        
        # Criterion 2b: 98% variance explained
        idx_98 = np.where(cumulative_variance >= 0.98)[0]
        if len(idx_98) > 0:
            n_dims_suggestions['98%_variance'] = idx_98[0] + 1
        
        # Criterion 3: Elbow method (largest drop in explained variance)
        if len(explained_variance_ratio) > 5:
            # Find the point where the explained variance drops below a threshold
            # or where the rate of decrease slows down significantly
            ratios = explained_variance_ratio[1:] / explained_variance_ratio[:-1]
            elbow_candidates = []
            
            # Find points where the ratio jumps significantly (indicating an elbow)
            for i in range(1, min(30, len(ratios))):
                if ratios[i] < 0.7:  # 30% drop from previous component
                    elbow_candidates.append(i + 1)
            
            if elbow_candidates:
                n_dims_suggestions['elbow_method'] = elbow_candidates[0]
        
        # Criterion 4: Kaiser criterion (eigenvalues > 1/n_features)
        # In PCA, this translates to explained variance > 1/n_features
        kaiser_threshold = 1.0 / X_np.shape[1]
        idx_kaiser = np.where(explained_variance_ratio >= kaiser_threshold)[0]
        if len(idx_kaiser) > 0:
            n_dims_suggestions['kaiser_criterion'] = len(idx_kaiser)
        
        # Criterion 5: Fixed percentage of original dimensions
        n_dims_suggestions['50%_original'] = max(10, X_np.shape[1] // 2)
        n_dims_suggestions['25%_original'] = max(5, X_np.shape[1] // 4)
        
        # Print suggestions
        print("\nDimensionality reduction suggestions:")
        for criterion, n_dims in n_dims_suggestions.items():
            variance_at_n = cumulative_variance[n_dims - 1] if n_dims <= len(cumulative_variance) else 1.0
            print(f"  {criterion}: {n_dims} dims (explains {variance_at_n:.1%} variance)")
        
        # Choose the best suggestion (prioritize 98% variance explained)
        if '98%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['98%_variance']
            reason = "98% variance explained"
        elif '95%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['95%_variance']
            reason = "95% variance explained"
        elif '90%_variance' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['90%_variance']
            reason = "90% variance explained"
        elif 'elbow_method' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['elbow_method']
            reason = "elbow method"
        elif '25%_original' in n_dims_suggestions:
            recommended_dims = n_dims_suggestions['25%_original']
            reason = "25% of original dimensions"
        else:
            recommended_dims = max(10, min(50, X_np.shape[1] // 4))
            reason = "conservative default"
        
        print(f"\nRecommended: {recommended_dims} dimensions ({reason})")
        print(f"This would explain {cumulative_variance[recommended_dims - 1]:.1%} of the variance")
        
        # Log dimensionality analysis metrics to wandb
        if wandblogger is not None:
            try:
                import wandb
                
                # Log all suggestions
                wandb_metrics = {}
                for criterion, n_dims in n_dims_suggestions.items():
                    variance_at_n = cumulative_variance[n_dims - 1] if n_dims <= len(cumulative_variance) else 1.0
                    wandb_metrics[f'dim_analysis/{criterion}_dims'] = n_dims
                    wandb_metrics[f'dim_analysis/{criterion}_variance'] = variance_at_n
                
                # Log recommended choice
                wandb_metrics.update({
                    'dim_analysis/recommended_dims': recommended_dims,
                    'dim_analysis/recommended_reason': reason,
                    'dim_analysis/recommended_variance': cumulative_variance[recommended_dims - 1],
                    'dim_analysis/original_dims': X_np.shape[1],
                    'dim_analysis/reduction_ratio': recommended_dims / X_np.shape[1],
                    'dim_analysis/max_components_analyzed': max_components,
                    'dim_analysis/first_pc_variance': explained_variance_ratio[0],
                    'dim_analysis/total_variance_in_first_10': np.sum(explained_variance_ratio[:10]) if len(explained_variance_ratio) >= 10 else np.sum(explained_variance_ratio),
                })
                
                # Log variance thresholds achieved
                for threshold in [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
                    idx_threshold = np.where(cumulative_variance >= threshold)[0]
                    if len(idx_threshold) > 0:
                        wandb_metrics[f'dim_analysis/dims_for_{int(threshold*100)}%_variance'] = idx_threshold[0] + 1
                
                wandblogger.log(wandb_metrics)
                print("✓ Dimensionality analysis metrics logged to wandb")
                
            except Exception as e:
                print(f"Warning: Could not log metrics to wandb: {e}")
        
        return recommended_dims



    def train_dimension_reducer(self, X, method='pca', n_components=None):
        """
        Train a dimensionality reduction model (PCA or UMAP)
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            method: 'pca' or 'umap'
            n_components: Number of components/dimensions to reduce to
            
        Returns:
            Trained reducer model and reduced features
        """
        print(f"Training {method.upper()} dimensionality reducer with {n_components} components...")
        
        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_np)
            
            # Print explained variance info
            explained_variance = np.sum(reducer.explained_variance_ratio_)
            print(f"PCA explained variance: {explained_variance:.1%}")
            
        elif method.lower() == 'umap':
            import umap #pip install umap-learn
            
            # UMAP parameters optimized for clustering
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                #random_state=42,
                n_jobs=-1,
                verbose=True
            )
            print(f"Fitting UMAP with {n_components} components...")
            X_reduced = reducer.fit_transform(X_np)
            reducer.verbose = False
            
            print(f"UMAP reduction complete: {X_np.shape[1]} -> {n_components} dimensions")
            
        else:
            raise ValueError(f"Unknown reduction method: {method}. Use 'pca' or 'umap'")
        
        # Convert back to tensor if needed
        if hasattr(X, 'device'):  # Original was a tensor
            import torch
            X_reduced = torch.tensor(X_reduced, dtype=self.dtype, device=X.device)
        
        return reducer, X_reduced


    def apply_dimension_reducer(self, X):
        """
        Apply trained dimensionality reducer to new data
        
        Args:
            X: Input features tensor/array of shape (N, input_features_dim)
            reducer: Trained PCA or UMAP reducer
            
        Returns:
            Reduced features
        """
        
        if self.dim_reducer is None:
            return X
        # Convert to numpy if needed
        if hasattr(X, 'cpu'):  # torch tensor
            X_np = X.cpu().numpy()
            original_device = X.device
            was_tensor = True
        else:
            X_np = X
            was_tensor = False
        
        # Apply reduction
        X_reduced = self.dim_reducer.transform(X_np)
        
        # Convert back to tensor if needed
        if was_tensor:
            import torch
            X_reduced = torch.tensor(X_reduced, dtype=self.dtype, device=original_device)
        
        return X_reduced


    def save_dimension_reducer(self, reducer, path_to_save):
        """
        Save dimensionality reducer to disk
        
        Args:
            reducer: Trained reducer model
            model_dir: Directory to save the model
            method: 'pca' or 'umap'
        """
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        with open(path_to_save, 'wb') as f:
            pickle.dump(reducer, f)
        self.reducer_path = path_to_save
        print(f"Reducer saved to: {path_to_save}")


    def load_dimension_reducer(self, reducer_path):
        """
        Load dimensionality reducer from disk
        
        Args:
            model_dir: Directory containing the saved model
            method: 'pca' or 'umap'
            
        Returns:
            Loaded reducer model
        """
        
        if not os.path.exists(reducer_path):
            raise FileNotFoundError(f"Reducer model not found at {reducer_path}. Please train and save it first.")
            
        print(f"Loading dimensionality reducer from: {reducer_path}")
        with open(reducer_path, 'rb') as f:
            reducer = pickle.load(f)
        print(f"Reducer loaded from: {reducer_path}")


        reducer.verbose = True
        
        return reducer


    def set_feature_names(self, feature_names):
        """
        Set names for the $self.input_features_dim phoneme features for better analysis
        
        Args:
            feature_names: List of feature names (length should be $self.input_features_dim)
        """
        if len(feature_names) != self.input_features_dim:
            raise ValueError(f"Expected {self.input_features_dim} feature names, got {len(feature_names)}")
        self.feature_names = feature_names
    

    def clustering_quality_analysis(self, features):
        """
        Analyze clustering quality using various metrics (optimized for large datasets)
        
        Args:
            features: Input features tensor of shape (N, input_features_dim)
            
        Returns:
            Dictionary containing clustering quality metrics
        """
        print("Calculating clustering quality metrics...")
        
        # Get cluster labels and distances
        labels, distances = self.model.predict(features, return_dists=True)
        
        # Convert to numpy for easier computation
        if isinstance(features, torch.Tensor):
            features_np = features.float().cpu().numpy()
        else:
            features_np = features
        
        labels_np = labels.cpu().numpy()
        distances_np = distances.cpu().numpy()
        centroids_np = self.model.cluster_centers_
        
        if isinstance(centroids_np, torch.Tensor):
            
            centroids_np = centroids_np.float().cpu().numpy()
        
        n_samples = len(features_np)
        print(f"Processing {n_samples} samples...")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # 1. Inertia (Within-Cluster Sum of Squares - WCSS)
        metrics['inertia'] = float(self.model.inertia_)
        print("✓ Inertia calculated")
        
        # 2. Fast Silhouette Score (using sampling for large datasets)
        try:
            from sklearn.metrics import silhouette_score, silhouette_samples
            
            if len(np.unique(labels_np)) > 1:  # Need at least 2 clusters
                # Use sampling for datasets larger than 10k samples
                if n_samples > 10000:
                    print(f"Large dataset detected ({n_samples} samples). Using sampling for silhouette score...")
                    sample_size = min(5000, n_samples // 2)  # Sample at most 5k points
                    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                    features_sample = features_np[sample_indices]
                    labels_sample = labels_np[sample_indices]
                    
                    silhouette_avg = silhouette_score(features_sample, labels_sample)
                    print(f"✓ Silhouette score calculated (sampled from {sample_size} points)")
                    
                    # Calculate per-cluster silhouette scores on sample
                    silhouette_scores_sample = silhouette_samples(features_sample, labels_sample)
                    cluster_silhouette = {}
                    for cluster_id in range(self.n_clusters):
                        cluster_mask = labels_sample == cluster_id
                        if np.sum(cluster_mask) > 0:
                            cluster_silhouette[cluster_id] = float(np.mean(silhouette_scores_sample[cluster_mask]))
                    
                    metrics['silhouette_score'] = float(silhouette_avg)
                    metrics['cluster_silhouette_scores'] = cluster_silhouette
                    metrics['silhouette_score_std'] = float(np.std(list(cluster_silhouette.values()))) if cluster_silhouette else 0.0
                    metrics['silhouette_sampled'] = True
                    metrics['silhouette_sample_size'] = sample_size
                else:
                    # Calculate on full dataset for smaller datasets
                    print("Calculating silhouette score on full dataset...")
                    silhouette_avg = silhouette_score(features_np, labels_np)
                    
                    # Calculate per-cluster silhouette scores
                    silhouette_scores = silhouette_samples(features_np, labels_np)
                    cluster_silhouette = {}
                    for cluster_id in range(self.n_clusters):
                        cluster_mask = labels_np == cluster_id
                        if np.sum(cluster_mask) > 0:
                            cluster_silhouette[cluster_id] = float(np.mean(silhouette_scores[cluster_mask]))
                    
                    metrics['silhouette_score'] = float(silhouette_avg)
                    metrics['cluster_silhouette_scores'] = cluster_silhouette
                    metrics['silhouette_score_std'] = float(np.std(list(cluster_silhouette.values()))) if cluster_silhouette else 0.0
                    metrics['silhouette_sampled'] = False
                    print("✓ Silhouette score calculated (full dataset)")
            else:
                metrics['silhouette_score'] = 0.0
                metrics['cluster_silhouette_scores'] = {}
                metrics['silhouette_score_std'] = 0.0
                metrics['silhouette_sampled'] = False
                
        except ImportError:
            print("Warning: scikit-learn not available for silhouette score calculation")
            metrics['silhouette_score'] = None
            metrics['cluster_silhouette_scores'] = {}
            metrics['silhouette_score_std'] = None
            metrics['silhouette_sampled'] = False
        
        # 3. Calinski-Harabasz Index (Variance Ratio Criterion) - Fast calculation
        try:
            from sklearn.metrics import calinski_harabasz_score
            if len(np.unique(labels_np)) > 1:
                # Use sampling for very large datasets
                if n_samples > 50000:
                    sample_size = min(10000, n_samples // 2)
                    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                    ch_score = calinski_harabasz_score(features_np[sample_indices], labels_np[sample_indices])
                    print(f"✓ Calinski-Harabasz score calculated (sampled from {sample_size} points)")
                else:
                    ch_score = calinski_harabasz_score(features_np, labels_np)
                    print("✓ Calinski-Harabasz score calculated")
                metrics['calinski_harabasz_score'] = float(ch_score)
            else:
                metrics['calinski_harabasz_score'] = 0.0
        except ImportError:
            metrics['calinski_harabasz_score'] = None
        
        # 4. Davies-Bouldin Index - Fast calculation
        try:
            from sklearn.metrics import davies_bouldin_score
            if len(np.unique(labels_np)) > 1:
                # Use sampling for very large datasets
                if n_samples > 50000:
                    sample_size = min(10000, n_samples // 2)
                    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                    db_score = davies_bouldin_score(features_np[sample_indices], labels_np[sample_indices])
                    print(f"✓ Davies-Bouldin score calculated (sampled from {sample_size} points)")
                else:
                    db_score = davies_bouldin_score(features_np, labels_np)
                    print("✓ Davies-Bouldin score calculated")
                metrics['davies_bouldin_score'] = float(db_score)
            else:
                metrics['davies_bouldin_score'] = 0.0
        except ImportError:
            metrics['davies_bouldin_score'] = None
        
        # 5. Fast Inter-cluster and Intra-cluster distances using pre-computed distances
        print("Calculating distance metrics...")
        
        # Use the already computed distances to centroids
        # Intra-cluster distances: distance from each point to its assigned centroid
        intra_cluster_distances = []
        for i, cluster_id in enumerate(labels_np):
            intra_cluster_distances.append(distances_np[i, cluster_id])
        
        # Inter-cluster distances: distances between centroids
        inter_cluster_distances = []
        if len(centroids_np) > 1:
            for i in range(len(centroids_np)):
                for j in range(i + 1, len(centroids_np)):
                    dist = np.linalg.norm(centroids_np[i] - centroids_np[j])
                    inter_cluster_distances.append(dist)
        
        # Store distance statistics
        if intra_cluster_distances:
            metrics['avg_intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
            metrics['std_intra_cluster_distance'] = float(np.std(intra_cluster_distances))
            metrics['min_intra_cluster_distance'] = float(np.min(intra_cluster_distances))
            metrics['max_intra_cluster_distance'] = float(np.max(intra_cluster_distances))
        else:
            metrics['avg_intra_cluster_distance'] = 0.0
            metrics['std_intra_cluster_distance'] = 0.0
            metrics['min_intra_cluster_distance'] = 0.0
            metrics['max_intra_cluster_distance'] = 0.0
        
        if inter_cluster_distances:
            metrics['avg_inter_cluster_distance'] = float(np.mean(inter_cluster_distances))
            metrics['std_inter_cluster_distance'] = float(np.std(inter_cluster_distances))
            metrics['min_inter_cluster_distance'] = float(np.min(inter_cluster_distances))
            metrics['max_inter_cluster_distance'] = float(np.max(inter_cluster_distances))
        else:
            metrics['avg_inter_cluster_distance'] = 0.0
            metrics['std_inter_cluster_distance'] = 0.0
            metrics['min_inter_cluster_distance'] = 0.0
            metrics['max_inter_cluster_distance'] = 0.0
        
        print("✓ Distance metrics calculated")
        
        # 6. Dunn Index (ratio of minimum inter-cluster distance to maximum intra-cluster distance)
        if inter_cluster_distances and intra_cluster_distances:
            dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
            metrics['dunn_index'] = float(dunn_index)
        else:
            metrics['dunn_index'] = 0.0
        
        # 7. Fast Cluster balance metrics
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        total_samples = len(labels_np)
        
        # Calculate cluster size statistics
        metrics["train_total_samples"] = int(total_samples)
        metrics['cluster_size_mean'] = float(np.mean(counts))
        metrics['cluster_size_std'] = float(np.std(counts))
        metrics['cluster_size_min'] = int(np.min(counts))
        metrics['cluster_size_max'] = int(np.max(counts))
        metrics['cluster_size_range'] = int(np.max(counts) - np.min(counts))
        
        # Calculate cluster balance ratio (std/mean)
        metrics['cluster_balance_ratio'] = float(np.std(counts) / np.mean(counts))
        
        # Calculate percentage of samples in largest and smallest clusters
        metrics['largest_cluster_percentage'] = float(np.max(counts) / total_samples * 100)
        metrics['smallest_cluster_percentage'] = float(np.min(counts) / total_samples * 100)
        
        print("✓ Cluster balance metrics calculated")
        
        # 8. Effective number of clusters (based on cluster sizes)
        cluster_proportions = counts / total_samples
        # Shannon entropy of cluster distribution
        shannon_entropy = -np.sum(cluster_proportions * np.log2(cluster_proportions + 1e-10))
        effective_clusters = 2 ** shannon_entropy
        metrics['effective_num_clusters'] = float(effective_clusters)
        metrics['cluster_entropy'] = float(shannon_entropy)
        
        # 9. Fast Cluster separation ratio using pre-computed distances
        print("Calculating separation ratios...")
        separation_ratios = []
        
        # Sample for large datasets to avoid O(n²) complexity
        sample_size_sep = min(5000, n_samples)
        if n_samples > sample_size_sep:
            sample_indices = np.random.choice(n_samples, sample_size_sep, replace=False)
        else:
            sample_indices = np.arange(n_samples)
        
        for idx in sample_indices:
            label = labels_np[idx]
            # Distance to own cluster center
            own_cluster_dist = distances_np[idx, label]
            
            # Distance to nearest other cluster center
            other_cluster_dists = distances_np[idx, :]
            other_cluster_dists = np.delete(other_cluster_dists, label)  # Remove own cluster
            nearest_other_dist = np.min(other_cluster_dists)
            
            if own_cluster_dist > 0:
                separation_ratio = nearest_other_dist / own_cluster_dist
                separation_ratios.append(separation_ratio)
        
        if separation_ratios:
            metrics['avg_separation_ratio'] = float(np.mean(separation_ratios))
            metrics['std_separation_ratio'] = float(np.std(separation_ratios))
            metrics['min_separation_ratio'] = float(np.min(separation_ratios))
            metrics['max_separation_ratio'] = float(np.max(separation_ratios))
            if n_samples > sample_size_sep:
                metrics['separation_ratio_sampled'] = True
                metrics['separation_sample_size'] = sample_size_sep
            else:
                metrics['separation_ratio_sampled'] = False
        else:
            metrics['avg_separation_ratio'] = 0.0
            metrics['std_separation_ratio'] = 0.0
            metrics['min_separation_ratio'] = 0.0
            metrics['max_separation_ratio'] = 0.0
            metrics['separation_ratio_sampled'] = False
        
        print("✓ Separation ratios calculated")
        
        # 10. Bayesian Information Criterion (BIC)
        print("Calculating BIC...")
        
        # BIC = n * ln(WCSS/n) + k * ln(n)
        # where n = number of samples, k = number of clusters, WCSS = within-cluster sum of squares
        n = n_samples
        k = self.n_clusters
        wcss = metrics['inertia']
        
        if wcss > 0 and n > 0:
            bic = n * np.log(wcss / n) + k * np.log(n)
            metrics['bic'] = float(bic)
            
            # Also calculate AIC for comparison
            # AIC = n * ln(WCSS/n) + 2*k
            aic = n * np.log(wcss / n) + 2 * k
            metrics['aic'] = float(aic)
            
            # Calculate BIC per sample for easier comparison across different dataset sizes
            metrics['bic_per_sample'] = float(bic / n)
            metrics['aic_per_sample'] = float(aic / n)
            
            print(f"✓ BIC calculated: {bic:.2f}")
        else:
            metrics['bic'] = float('inf')
            metrics['aic'] = float('inf')
            metrics['bic_per_sample'] = float('inf')
            metrics['aic_per_sample'] = float('inf')
            print("✓ BIC calculation skipped (invalid WCSS)")
        
        print("Clustering quality analysis complete.")
        return metrics
    

    def make_clusters_visualization(self, wandblogger, features, labels, method="umap", n_components=2, n_neighbors=15, min_dist=0.1, ):
        """
        Create UMAP visualization of the clustered features
        
        Args:
            features: Input features tensor/array of shape (N, input_features_dim)
            labels: Cluster labels tensor/array of shape (N,)
            method: Dimensionality reduction method ('pca' or 'umap')
            n_components: Number of dimensions for UMAP (default 2)
            n_neighbors: Number of neighbors for UMAP (default 15)
            min_dist: Minimum distance between points in UMAP (default 0.1)
            
        Returns:
            UMAP-reduced features and plot
        """
        import matplotlib.pyplot as plt
        
        # Convert to numpy if needed
        if hasattr(features, 'cpu'):  # torch tensor
            features_np = features.cpu().numpy()
        else:
            features_np = features

        if hasattr(labels, 'cpu'):  # torch tensor
            labels = labels.cpu().numpy()

        if method.lower() == 'umap':        
            print("Creating UMAP visualization...")
            import umap
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_jobs=1)
            reduced_features = reducer.fit_transform(features_np)   
            print(f"UMAP reduction complete: {features_np.shape[1]} -> {n_components} dimensions")
        else: # pca
            from sklearn.decomposition import PCA
            
            print("Creating PCA visualization...")
            reducer = PCA(n_components=n_components)
            reduced_features = reducer.fit_transform(features_np)
            print(f"PCA reduction complete: {features_np.shape[1]} -> {n_components} dimensions")

        # Create scatter plot
        # Use a colormap for better visualization
        if (n_components == 2):
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.7)
            plt.colorbar(scatter, label='Cluster Label')
            plt.title('UMAP Visualization of Clusters')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.grid(True, alpha=0.3)
        elif (n_components == 3):
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels, cmap='Spectral', s=5, alpha=0.7)
            plt.colorbar(scatter, label='Cluster Label')
            ax.set_title('UMAP Visualization of Clusters')
            ax.set_xlabel('UMAP Component 1')
            ax.set_ylabel('UMAP Component 2')
            ax.set_zlabel('UMAP Component 3')
        else:
            raise ValueError("UMAP visualization only supports 2 or 3 components")
        
        # Save the plot
        plot_path = os.path.join(self.model_dir, 'clusters_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"UMAP visualization saved to: {plot_path}")

        import wandb
        wandblogger.log({"clusters_visualization": wandb.Image(plt)})
        print("✓ Clusters visualization logged to wandb")
        plt.close()


    def train_log(self, total_samples, cluster_stats, importance_data, quality_data, clustering_model_dir, vis_features=None, vis_labels=None, wandblogger=None):
        """
        Log training results to file and wandb
        
        Args:
            cluster_stats: List of strings containing cluster distribution statistics
            importance_data: Dictionary containing feature importance analysis
            quality_data: Dictionary containing clustering quality metrics
            clustering_model_dir: Directory where the model is saved
            vis_features: Optional features for visualization (e.g., PCA/UMAP reduced features)
            vis_labels: Optional labels for visualization
            wandblogger: Optional wandb logger object
        """
        report_path = os.path.join(clustering_model_dir, 'report.txt')
        
        # Log to wandb if provided
        if wandblogger is not None:
            try:
                import wandb
                # Log main quality metrics
                wandb_metrics = {
                    'train_total_samples': total_samples,
                    'silhouette_score': quality_data.get('silhouette_score', 0),
                    'calinski_harabasz_score': quality_data.get('calinski_harabasz_score', 0),
                    'davies_bouldin_score': quality_data.get('davies_bouldin_score', 0),
                    'dunn_index': quality_data.get('dunn_index', 0),
                    'inertia': quality_data.get('inertia', 0),
                    'bic': quality_data.get('bic', 0),
                    'aic': quality_data.get('aic', 0),
                    'bic_per_sample': quality_data.get('bic_per_sample', 0),
                    'aic_per_sample': quality_data.get('aic_per_sample', 0),
                    'effective_num_clusters': quality_data.get('effective_num_clusters', 0),
                    'cluster_entropy': quality_data.get('cluster_entropy', 0),
                    'cluster_balance_ratio': quality_data.get('cluster_balance_ratio', 0),
                    'avg_separation_ratio': quality_data.get('avg_separation_ratio', 0),
                    'avg_intra_cluster_distance': quality_data.get('avg_intra_cluster_distance', 0),
                    'avg_inter_cluster_distance': quality_data.get('avg_inter_cluster_distance', 0),
                    'cluster_size_min': quality_data.get('cluster_size_min', 0),
                    'cluster_size_max': quality_data.get('cluster_size_max', 0),
                    'cluster_size_mean': quality_data.get('cluster_size_mean', 0),
                    'cluster_size_std': quality_data.get('cluster_size_std', 0),
                    'largest_cluster_percentage': quality_data.get('largest_cluster_percentage', 0),
                    'smallest_cluster_percentage': quality_data.get('smallest_cluster_percentage', 0),
                }
                
                # Add model configuration
                wandb_metrics.update({
                    'config/n_clusters': self.n_clusters,
                    'config/input_features_dim': self.input_features_dim,
                    'config/device': str(self.device),
                })

                if self.reduced_dims is not None:
                    wandb_metrics['config/reduced_dims'] = self.reduced_dims
                    if self.dim_reducer is not None:
                        wandb_metrics['config/dimensionality_reducer'] = type(self.dim_reducer).__name__
                else:
                    wandb_metrics['config/reduced_dims'] = -1
                    wandb_metrics['config/dimensionality_reducer'] = 'None'
                
                # Log feature importance (top 10)
                if importance_data.get("global_importance"):
                    for i, (feature, importance) in enumerate(list(importance_data["global_importance"].items())[:10]):
                        wandb_metrics[f'feature_importance/top_{i+1}_{feature}'] = importance
                
                # Log per-cluster silhouette scores (best and worst 5)
                if quality_data.get('cluster_silhouette_scores'):
                    cluster_scores = quality_data['cluster_silhouette_scores']
                    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Best 5 clusters
                    for i, (cluster_id, score) in enumerate(sorted_clusters[:3]):
                        wandb_metrics[f'silhouette/best_{i}_sil'] = score
                        #wandb_metrics[f'cluster_silhouette/best_cluster_id'] = cluster_id
                    
                    # Worst 5 clusters
                    for i, (cluster_id, score) in enumerate(sorted_clusters[-3:]):
                        wandb_metrics[f'silhouette/worst_{i}_sil'] = score
                        #wandb_metrics[f'cluster_silhouette/worst_{i}_id'] = cluster_id
                
                wandblogger.log(wandb_metrics)
                print("✓ Metrics logged to wandb")
                
                # Create and log feature importance plot if available
                if importance_data.get("visualization_data"):
                    try:
                        import matplotlib.pyplot as plt
                        viz_data = importance_data["visualization_data"]
                        top_n = min(40, len(viz_data["importance_values"]))
                        top_indices = np.argsort(viz_data["importance_values"])[::-1][:top_n]
                        
                        plt.figure(figsize=(12, 8))
                        plt.barh(range(top_n), [viz_data["importance_values"][i] for i in top_indices])
                        plt.yticks(range(top_n), [viz_data["feature_names"][i] for i in top_indices])
                        plt.xlabel('Feature Importance')
                        plt.title(f'Top {top_n} Most Important Features')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        
                        wandblogger.log({"feature_importance_plot": wandb.Image(plt)})
                        plt.close()
                        print("✓ Feature importance plot logged to wandb")
                    except Exception as e:
                        print(f"Warning: Could not create feature importance plot: {e}")
                
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")
        
        # Generate text report
        with open(report_path, 'w') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write("PHONEME CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Write basic model information
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"Number of clusters: {self.n_clusters}\n")
            f.write(f"Input features dimension: {self.input_features_dim}\n")
            if self.reduced_dims is not None:
                f.write(f"Reduced dimensions: {self.reduced_dims}\n")
                if self.dim_reducer is not None:
                    f.write(f"Dimensionality reduction method: {type(self.dim_reducer).__name__}\n")
            else:
                f.write("Dimensionality reduction: None\n")
                

            
            
            f.write(f"Device: {self.device}\n")
            
            f.write(f"Total training samples: {total_samples}\n\n")
            
            # Write cluster distribution
            f.write("CLUSTER DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for line in cluster_stats:
                f.write(line + "\n")
            f.write("\n")
            
            # Write clustering quality metrics
            f.write("CLUSTERING QUALITY METRICS:\n")
            f.write("-" * 30 + "\n")
            
            # Overall quality scores
            f.write("Overall Quality Scores:\n")
            if quality_data.get('silhouette_score') is not None:
                f.write(f"  Silhouette Score: {quality_data['silhouette_score']:.4f} (range: -1 to 1, higher is better)\n")
                if quality_data.get('silhouette_score_std') is not None:
                    f.write(f"  Silhouette Score Std: {quality_data['silhouette_score_std']:.4f}\n")
            
            if quality_data.get('calinski_harabasz_score') is not None:
                f.write(f"  Calinski-Harabasz Score: {quality_data['calinski_harabasz_score']:.2f} (higher is better)\n")
            
            if quality_data.get('davies_bouldin_score') is not None:
                f.write(f"  Davies-Bouldin Score: {quality_data['davies_bouldin_score']:.4f} (lower is better)\n")
            
            f.write(f"  Inertia (WCSS): {quality_data['inertia']:.2f} (lower is better)\n")
            f.write(f"  Dunn Index: {quality_data['dunn_index']:.4f} (higher is better)\n")
            
            # Information Criteria
            f.write(f"  BIC: {quality_data['bic']:.2f} (lower is better)\n")
            f.write(f"  AIC: {quality_data['aic']:.2f} (lower is better)\n")
            f.write(f"  BIC per sample: {quality_data['bic_per_sample']:.4f}\n")
            f.write(f"  AIC per sample: {quality_data['aic_per_sample']:.4f}\n")
            
            # Distance metrics
            f.write("\nDistance Metrics:\n")
            f.write(f"  Avg Intra-cluster Distance: {quality_data['avg_intra_cluster_distance']:.4f} ± {quality_data['std_intra_cluster_distance']:.4f}\n")
            f.write(f"  Avg Inter-cluster Distance: {quality_data['avg_inter_cluster_distance']:.4f} ± {quality_data['std_inter_cluster_distance']:.4f}\n")
            f.write(f"  Min Inter-cluster Distance: {quality_data['min_inter_cluster_distance']:.4f}\n")
            f.write(f"  Max Intra-cluster Distance: {quality_data['max_intra_cluster_distance']:.4f}\n")
            
            # Separation metrics
            f.write("\nCluster Separation Metrics:\n")
            f.write(f"  Avg Separation Ratio: {quality_data['avg_separation_ratio']:.4f} ± {quality_data['std_separation_ratio']:.4f}\n")
            f.write(f"  Min/Max Separation Ratio: {quality_data['min_separation_ratio']:.4f} / {quality_data['max_separation_ratio']:.4f}\n")
            
            # Cluster balance metrics
            f.write("\nCluster Balance Metrics:\n")
            f.write(f"  Cluster Size Range: {quality_data['cluster_size_min']} - {quality_data['cluster_size_max']} samples\n")
            f.write(f"  Cluster Size Mean ± Std: {quality_data['cluster_size_mean']:.1f} ± {quality_data['cluster_size_std']:.1f}\n")
            f.write(f"  Cluster Balance Ratio: {quality_data['cluster_balance_ratio']:.4f} (lower is more balanced)\n")
            f.write(f"  Largest Cluster: {quality_data['largest_cluster_percentage']:.2f}% of data\n")
            f.write(f"  Smallest Cluster: {quality_data['smallest_cluster_percentage']:.2f}% of data\n")
            f.write(f"  Effective Number of Clusters: {quality_data['effective_num_clusters']:.2f}\n")
            f.write(f"  Cluster Entropy: {quality_data['cluster_entropy']:.4f}\n")
            
            # Per-cluster silhouette scores (top 10 and bottom 5)
            if quality_data.get('cluster_silhouette_scores'):
                f.write("\nPer-Cluster Silhouette Scores:\n")
                cluster_scores = quality_data['cluster_silhouette_scores']
                sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
                
                f.write("  Best performing clusters:\n")
                for i, (cluster_id, score) in enumerate(sorted_clusters[:10]):
                    f.write(f"    Cluster {cluster_id}: {score:.4f}\n")
                
                if len(sorted_clusters) > 15:
                    f.write("  ...\n")
                
                f.write("  Worst performing clusters:\n")
                for cluster_id, score in sorted_clusters[-5:]:
                    f.write(f"    Cluster {cluster_id}: {score:.4f}\n")
            
            f.write("\n")
                    
            # Write feature importance analysis
            f.write("FEATURE IMPORTANCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write("Top 20 Most Important Features:\n")
            for i, (feature, importance) in enumerate(list(importance_data["global_importance"].items())[:20]):
                f.write(f"  {i+1:2d}. {feature}: {importance:.4f}\n")
            
            f.write("\nFeature Importance by Cluster (top 10 clusters):\n")
            for i, (cluster, features) in enumerate(list(importance_data["cluster_importance"].items())[:10]):
                f.write(f"\n  {cluster}:\n")
                for j, (feature, importance) in enumerate(features.items()):
                    f.write(f"    {j+1}. {feature}: {importance:.4f}\n")
            
            # Add quality assessment summary
            f.write("\n")
            f.write("QUALITY ASSESSMENT SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            # Provide interpretation of metrics
            quality_score = 0
            quality_comments = []
            
            if quality_data.get('silhouette_score') is not None:
                sil_score = quality_data['silhouette_score']
                if sil_score > 0.5:
                    quality_comments.append("Excellent cluster separation (Silhouette > 0.5)")
                    quality_score += 2
                elif sil_score > 0.3:
                    quality_comments.append("Good cluster separation (Silhouette > 0.3)")
                    quality_score += 1
                elif sil_score > 0.1:
                    quality_comments.append("Fair cluster separation (Silhouette > 0.1)")
                else:
                    quality_comments.append("Poor cluster separation (Silhouette ≤ 0.1)")
                    quality_score -= 1
            
            # Check cluster balance
            balance_ratio = quality_data['cluster_balance_ratio']
            if balance_ratio < 0.5:
                quality_comments.append("Well-balanced cluster sizes")
                quality_score += 1
            elif balance_ratio < 1.0:
                quality_comments.append("Moderately balanced cluster sizes")
            else:
                quality_comments.append("Imbalanced cluster sizes")
                quality_score -= 1
            
            # Check effective clusters
            effective_ratio = quality_data['effective_num_clusters'] / self.n_clusters
            if effective_ratio > 0.8:
                quality_comments.append("Most clusters are effectively utilized")
                quality_score += 1
            elif effective_ratio > 0.6:
                quality_comments.append("Good cluster utilization")
            else:
                quality_comments.append("Many clusters may be underutilized")
                quality_score -= 1
            
            # Overall assessment
            if quality_score >= 3:
                overall_quality = "EXCELLENT"
            elif quality_score >= 1:
                overall_quality = "GOOD"
            elif quality_score >= -1:
                overall_quality = "FAIR"
            else:
                overall_quality = "POOR"
            
            f.write(f"Overall Quality Assessment: {overall_quality}\n\n")
            f.write("Key Observations:\n")
            for comment in quality_comments:
                f.write(f"  • {comment}\n")
            
            # Add recommendations
            f.write("\nRecommendations:\n")
            if quality_data.get('silhouette_score', 0) < 0.3:
                f.write("  • Consider adjusting the number of clusters for better separation\n")
            if balance_ratio > 1.0:
                f.write("  • Consider using different initialization or balancing techniques\n")
            if effective_ratio < 0.6:
                f.write("  • Consider reducing the number of clusters to improve utilization\n")
            if quality_data['dunn_index'] < 1.0:
                f.write("  • Clusters may be too close together - consider feature selection or dimensionality reduction\n")
            
            # BIC-based recommendations
            bic_per_sample = quality_data.get('bic_per_sample', 0)
            if bic_per_sample > 15:  # Threshold can be adjusted based on your data
                f.write("  • High BIC suggests too many clusters - consider reducing the number of clusters\n")
            elif bic_per_sample < 5:
                f.write("  • Low BIC suggests good model fit for the current number of clusters\n")
            
            f.write(f"\nModel Selection Notes:\n")
            f.write(f"  • BIC favors simpler models and penalizes overfitting\n")
            f.write(f"  • Compare BIC values across different cluster numbers to find optimal k\n")
            f.write(f"  • Lower BIC indicates better balance between fit and complexity\n")
        
        print(f"Report saved to {report_path}")
        if wandblogger is not None and (vis_features is not None and vis_labels is not None):
            try:
                self.make_clusters_visualization(wandblogger, vis_features, vis_labels)
            except Exception as e:
                print(f"Warning: Could not create UMAP visualization: {e}")



    
    # Modified train method (key changes only)
    def train(self, features_collation, clustering_model_dir, normalize=True, max_iter=1000, dim_reducer_method='pca', reduced_dims=None, wandblogger=None):
        """
        Modified train method with dimensionality reduction support
        
        Args:
            features_collation: Object containing phoneme features
            clustering_model_dir: Directory to save the clustering model
            max_iter: Maximum number of iterations for K-means
            dim_reducer_method: 'pca' or 'umap'
            reduced_dims: Number of dimensions to reduce to (0=auto, None=no reduction)
        """
        self.model_dir = clustering_model_dir
        print(f"model_dir: {self.model_dir}")
        if not os.path.exists(clustering_model_dir):
            os.makedirs(clustering_model_dir)

        # Get features
        features = features_collation.features
        original_dims = features.shape[1]
        
        # Handle dimensionality reduction
        if reduced_dims is not None:

            assert (original_dims > reduced_dims*1.1), "Original dimensions must be greater than reduced_dims by at least 10%. Either set reduced_dims to None to disable dimensionality reduction, or descrease reduced_dims to a smaller value."
            
            # split by _u
            reducer_path_dir = '_u'+os.path.basename( clustering_model_dir).split('_u')[1]
            reducer_path = os.path.join(os.path.dirname(clustering_model_dir), 'reducers', reducer_path_dir+""+ dim_reducer_method+"_"+str(reduced_dims)+"_len"+str(len(features))+"_"+str(features.shape[1])+'.pkl')
            
            if os.path.exists(reducer_path):
                print(f"Loading existing dimensionality reducer from {reducer_path}")
                self.dim_reducer = self.load_dimension_reducer(reducer_path)
                print("Transforming features with loaded dimensionality reducer...")
                features_reduced = self.dim_reducer.transform(features)
                reduced_dims = features_reduced.shape[1]
                print(f"Features reduced to : {reduced_dims} dimensions")

            else:

                print(f"Training new dimensionality reducer using {dim_reducer_method}...")
                os.makedirs(os.path.dirname(reducer_path), exist_ok=True)
                eigen_plot_path = reducer_path + '.eigen.png'
                if reduced_dims == 0:
                    # Auto-determine optimal dimensions
                    auto_dims_suggestion = self.auto_dims_eigen_plot(features, eigen_plot_path, wandblogger)
                    reduced_dims = auto_dims_suggestion
                
                
                # Train dimensionality reducer
                self.dim_reducer, features_reduced = self.train_dimension_reducer(features, method=dim_reducer_method, n_components=reduced_dims)
                
                # Save the reducer
                self.save_dimension_reducer(self.dim_reducer, reducer_path)
                
            # Update features for clustering
            features = features_reduced
            self.reduced_dims = reduced_dims
            
            print(f"Dimensionality reduced: {original_dims} -> {reduced_dims}")
            
            # Save reduction info for later use
            reduction_info = {
                'method': dim_reducer_method,
                'original_dims': int(original_dims),
                'reduced_dims': int(reduced_dims),
                'reduction_ratio': float(reduced_dims) / float(original_dims),
                'reducer_path': reducer_path,
            }
            
            # Add explained variance for PCA
            if dim_reducer_method.lower() == 'pca' and hasattr(self.dim_reducer, 'explained_variance_ratio_'):
                reduction_info['explained_variance'] = float(np.sum(self.dim_reducer.explained_variance_ratio_))
            
            with open(os.path.join(clustering_model_dir, 'reduction_info.json'), 'w') as f:
                json.dump(reduction_info, f, indent=2)
        else:
            self.dim_reducer = None
            self.reduced_dims = None
            reduction_info = None
        
        # Train K-means clustering model on (possibly reduced) features
        print("Training K-means clustering model...")
        
        self.model.fit(features, normalize=normalize, max_iter=max_iter)
        
        # Save model
        self.model.save_model(clustering_model_dir)
        
        print(f"Training complete. Model saved to {clustering_model_dir}")
        

        # Calculate cluster statistics, feature importance, and quality metrics
        cluster_stats = self._get_cluster_stats(features)
        importance_data = self._analyze_feature_importance()
        quality_data = self.clustering_quality_analysis(features)
        
        total_samples = len(features_collation.features) if hasattr(features_collation, 'features') else 0
        vis_features = features[:10000]
        vis_labels, _dists = self.model.predict(vis_features)

        self.train_log(total_samples, cluster_stats, importance_data, quality_data, clustering_model_dir, vis_features, vis_labels, wandblogger=wandblogger)
        
        return self 
        
    def _get_cluster_stats(self, features):
        """Return statistics about cluster distribution as a list of strings, including min/max/mean/std and percentages"""
        labels, dists = self.model.predict(features)

        unique_labels, counts = np.unique(labels.cpu().numpy(), return_counts=True)
        total = len(labels)
        lines = []
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            if i < 10 or i >= len(unique_labels) - 5:  # First 10 and last 5 clusters
                lines.append(f"Cluster {label}: {count} samples ({count/total*100:.2f}%)")
            elif i == 10:
                lines.append("...")

        # Add min, max, mean, std of cluster memberships
        min_count = np.min(counts)
        max_count = np.max(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_percent = min_count / total * 100
        max_percent = max_count / total * 100

        lines.append("")
        lines.append(f"Total samples: {total}")
        lines.append(f"Cluster membership statistics:")
        lines.append(f"  Min:  {min_count} ({min_percent:.2f}%)")
        lines.append(f"  Max:  {max_count} ({max_percent:.2f}%)")
        lines.append(f"  Mean: {mean_count:.2f}")
        lines.append(f"  Std:  {std_count:.2f}")
        return lines

    def _analyze_feature_importance(self):
        """Analyze and return feature importance information"""
        if self.feature_names is None:
            # Create default feature names
            self.feature_names = [f"Feature_{i}" for i in range(self.input_features_dim)]
        importance_data = self.model.analyze_features(self.feature_names)
        return importance_data
    
    

    # Modified load_model method
    def load_model(self, clustering_model_dir):
        """
        Modified load_model method with dimensionality reduction support
        """
        self.model_dir = clustering_model_dir
        self.model = GPUKMeans.load_model(clustering_model_dir, device=self.device)
        self.n_clusters = self.model.n_clusters
        #parent_dir = os.path.dirname(clustering_model_dir)
        #dim_reducer_path = os.path.join(parent_dir, 'reducers', 'dimensionality_reducer.pt')
        # Load reduction info if available
        reduction_info_path = os.path.join(clustering_model_dir, 'reduction_info.json')
        if os.path.exists(reduction_info_path):
            import json
            with open(reduction_info_path, 'r') as f:
                reduction_info = json.load(f)
            
            method = reduction_info['method']
            self.reduced_dims = reduction_info['reduced_dims']
            self.reducer_path = reduction_info['reducer_path']
            # Load the dimensionality reducer
            self.dim_reducer = self.load_dimension_reducer(self.reducer_path)
            
            if self.dim_reducer is not None:
                print(f"Loaded {method.upper()} reducer: {reduction_info['original_dims']} -> {reduction_info['reduced_dims']} dims")
                if 'explained_variance' in reduction_info:
                    print(f"Explained variance: {reduction_info['explained_variance']:.1%}")
        else:
            self.dim_reducer = None
            self.reduced_dims = None
        
        return self
      
    





if __name__ == "__main__":
    print("Starting  features clustering")
    torch.manual_seed(42)
    
    print("Done!")


