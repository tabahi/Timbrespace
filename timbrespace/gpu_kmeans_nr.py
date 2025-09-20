import os
import numpy as np
import json
import torch
import time
# no reducer fork


class CustomScaler:
    """
    PyTorch custom scaler with multiple normalization methods and bf16 support
    """
    def __init__(self, device="cuda:0", method="none", dtype=torch.float32):
        """
        Args:
            method: "none" for no normalization, "l1" for L1 normalization, 
                   "l2" for L2 normalization, "standard" for mean/std normalization
            device: Device to run on (e.g., "cuda:0" or "cpu")
            dtype: Data type for computations (torch.float32, torch.bfloat16, etc.)
        """
        self.device = device
        self.method = None if method == "none" else method
        self.dtype = dtype
        self.mean_ = None
        self.std_ = None
        self.fitted = False
        
        # Validate method
        valid_methods = [None, "l1", "l2", "standard"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")
        
    def fit(self, X):
        """
        Fit the scaler to the data
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
        """
        # Ensure X is on the correct device and dtype
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
            
        if self.method == "standard":
            # Use float32 for numerical stability during computation
            X_compute = X.float() if self.dtype == torch.bfloat16 else X
            
            self.mean_ = torch.mean(X_compute, dim=0, keepdim=True)
            self.std_ = torch.std(X_compute, dim=0, keepdim=True, unbiased=False)
            
            # Avoid division by zero
            self.std_ = torch.where(self.std_ == 0, torch.ones_like(self.std_), self.std_)
            
            # Convert back to target dtype if needed
            if self.dtype == torch.bfloat16:
                self.mean_ = self.mean_.to(self.dtype)
                self.std_ = self.std_.to(self.dtype)
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # For l1, l2, and none methods, we don't need to store anything during fit
        self.fitted = True
        return self
        
    def transform(self, X):
        """Transform with mixed precision support"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        # Ensure X is on the correct device and dtype
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        if self.method == "standard":
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Standard scaling requires fit() to be called first.")
            return (X - self.mean_) / self.std_
            
        elif self.method == "l1":
            # L1 normalization (each row sums to 1)
            norm = torch.sum(torch.abs(X), dim=1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            return X / norm
            
        elif self.method == "l2":
            # L2 normalization (each row has unit L2 norm)
            norm = torch.linalg.vector_norm(X, ord=2, dim=1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            return X / norm
            
        else:  # method is None
            # No normalization
            return X
    
    def fit_transform(self, X):
        """Fit the scaler and transform the data in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform with mixed precision support"""
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
            
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        if self.method == "standard":
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Standard scaling requires fit() to be called first.")
            return X * self.std_ + self.mean_
            
        elif self.method in ["l1", "l2"]:
            raise NotImplementedError(
                f"Inverse transform is not supported for {self.method} normalization "
                "without storing original norms. Consider using standard scaling."
            )
            
        else:  # method is None
            # No normalization was applied, so no inverse needed
            return X
            
    def get_params(self):
        """Get scaler parameters"""
        return {
            'method': self.method,
            'device': self.device,
            'dtype': self.dtype,
            'fitted': self.fitted,
            'mean_': self.mean_,
            'std_': self.std_
        }
    
    def __repr__(self):
        return (f"CustomScaler(method='{self.method}', device='{self.device}', "
                f"dtype={self.dtype}, fitted={self.fitted})")
   

class GPUKMeans:
    """
    K-means implementation with bf16 support
    """
    def __init__(self, n_clusters=50, tol=1e-5, random_state=42, device="cuda:0", dtype=torch.float32):
        self.n_clusters = n_clusters
        self.max_iter = 1000
        self.tol = tol
        self.random_state = random_state
        self.device = device # Device to run on (e.g., "cuda:0" or "cpu")
        self.dtype = dtype  # Support for bf16, options: torch.float32, torch.bfloat16, torch.float16
        self.cluster_centers_ = None
        self.inertia_ = None
        self.scaler = None
        self.feature_importance_ = None
        self.cluster_feature_importance_ = None

        # Validate bf16 support
        if dtype == torch.bfloat16:
            if not torch.cuda.is_available():
                raise ValueError("bf16 requires CUDA")
            if not torch.cuda.is_bf16_supported():
                print("Warning: bf16 may not be fully supported on this GPU")
        
    def _initialize_centroids(self, X):
        """Initialize centroids with proper dtype"""
        torch.manual_seed(self.random_state)
        indices = torch.randperm(X.shape[0], device=self.device)[:self.n_clusters]
        return X[indices].clone()

    def fit(self, X, normalize=True, max_iter=1000, use_mixed_precision=None):
        """
        Fit K-means clustering with bf16 support
        
        Args:
            X: Input features tensor
            normalize: Whether to normalize features
            max_iter: Maximum iterations
            use_mixed_precision: If True, use float32 for critical computations even with bf16 data
        """
        start_time = time.time()
        print(f"Starting K-means clustering with {self.n_clusters} clusters")
        print(f"Device: {self.device}, Data type: {self.dtype}")
        
        # Handle mixed precision default
        if use_mixed_precision is None:
            use_mixed_precision = (self.dtype == torch.bfloat16)
        
        # Convert input to target dtype and device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        
        print(f"Input data shape: {X.shape}, dtype: {X.dtype}")
        
        # Normalize features if requested
        if normalize:
            print("Normalizing features...")
            self.scaler = CustomScaler(device=self.device, method="l2", dtype=self.dtype)
            X = self.scaler.fit_transform(X)
            print("Normalization complete.")
        else:
            self.scaler = None
        
        # Initialize centroids
        centroids = self._initialize_centroids(X)
        
        # Iterative refinement
        prev_inertia = float('inf')
        self.max_iter = max_iter
        
        for iteration in range(self.max_iter):
            distances = torch.cdist(X, centroids, p=2.0) ** 2
            
            # Assign each point to the nearest centroid
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Calculate inertia
            inertia = torch.sum(torch.min(distances, dim=1).values)
            
            # Check for convergence
            centroid_shift = torch.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration+1}/{self.max_iter}, Inertia: {inertia.item():.4f}, Shift: {centroid_shift.item():.8f}")
            
            # Check for convergence
            if abs(prev_inertia - inertia.item()) < self.tol:
                print(f"Converged at iteration {iteration+1}")
                break
                
            prev_inertia = inertia.item()
        
        self.cluster_centers_ = centroids
        self.inertia_ = inertia.item()
        
        # Calculate feature importance (always in float32 for stability)
        self._calculate_feature_importance(X, labels, use_mixed_precision)
        
        elapsed_time = time.time() - start_time
        print(f"K-means clustering completed in {elapsed_time:.2f} seconds")
        return self
    
    
    def _calculate_feature_importance(self, X, labels, use_mixed_precision=True):
        """Calculate feature importance with mixed precision support"""
        print("Calculating feature importance...")
        
        centroids = self.cluster_centers_
        n_features = centroids.shape[1]
        
        # Always compute feature importance in float32 for numerical stability
        if self.dtype == torch.bfloat16:
            centroids_compute = centroids.float()
        else:
            centroids_compute = centroids
        
        # Global feature importance
        centroid_variance = torch.var(centroids_compute, dim=0)
        global_importance = centroid_variance / torch.sum(centroid_variance)
        
        # Per-cluster feature importance
        cluster_importance = torch.zeros((self.n_clusters, n_features), device=self.device, dtype=torch.float32)
        
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() == 0:
                continue
                
            other_centroids = torch.cat([centroids_compute[:k], centroids_compute[k+1:]], dim=0)
            if len(other_centroids) == 0:
                continue
                
            feature_dists = torch.zeros(n_features, device=self.device, dtype=torch.float32)
            for feature_idx in range(n_features):
                this_centroid_feature = centroids_compute[k, feature_idx]
                other_centroids_feature = other_centroids[:, feature_idx]
                feature_dist = torch.mean((this_centroid_feature - other_centroids_feature) ** 2)
                feature_dists[feature_idx] = feature_dist
            
            if torch.sum(feature_dists) > 0:
                cluster_importance[k] = feature_dists / torch.sum(feature_dists)
        
        # Store as numpy arrays
        self.feature_importance_ = global_importance.cpu().numpy()
        self.cluster_feature_importance_ = cluster_importance.cpu().numpy()
        
        # Print top features
        top_indices = torch.argsort(global_importance, descending=True)[:10].cpu().numpy()
        print("\nTop 10 most important features (global):")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: {global_importance[idx].item():.4f}")
    
    def analyze_features(self, feature_names=None):
        """
        Analyze feature importance and return detailed report
        
        Args:
            feature_names: Optional list of feature names for better readability
        
        Returns:
            Dictionary with feature importance analysis
        """
        if self.feature_importance_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        
        print("len self.feature_importance_:", len(self.feature_importance_))
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance_))]
        
        # Get indices sorted by importance
        sorted_indices = np.argsort(self.feature_importance_)[::-1]
        
        # Global feature importance
        global_importance = {
            feature_names[idx]: float(self.feature_importance_[idx])
            for idx in sorted_indices
        }
        
        # Per-cluster feature importance
        cluster_importance = {}
        for k in range(self.n_clusters):
            # Get top 5 features for this cluster
            cluster_sorted = np.argsort(self.cluster_feature_importance_[k])[::-1][:5]
            cluster_importance[f"Cluster_{k}"] = {
                feature_names[idx]: float(self.cluster_feature_importance_[k][idx])
                for idx in cluster_sorted
            }
        
        # Create visualization data for plotting
        visualization_data = {
            "feature_names": feature_names,
            "importance_values": self.feature_importance_.tolist()
        }
        
        return {
            "global_importance": global_importance,
            "cluster_importance": cluster_importance,
            "visualization_data": visualization_data
        }
    

    def predict(self, X, return_dists=False, use_mixed_precision=None):
            """Predict with bf16 support"""
            if use_mixed_precision is None:
                use_mixed_precision = (self.dtype == torch.bfloat16)
            
            # Convert to proper dtype and device
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=self.dtype, device=self.device)
            else:
                X = X.to(device=self.device, dtype=self.dtype)
            
            # Apply normalization
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Calculate distances
            centroids = self.cluster_centers_
            
            distances = torch.cdist(X, centroids, p=2.0)
            
            labels = torch.argmin(distances, dim=1)
            
            if return_dists:
                return labels, distances.float()
            else:
                return labels, None
    
    def save_model(self, model_dir):
        """Save model with dtype information"""
        os.makedirs(model_dir, exist_ok=True)
        
        centroids_cpu = self.cluster_centers_.float().cpu().numpy()
        
        model_data = {
            'cluster_centers': centroids_cpu.tolist(),
            'n_clusters': self.n_clusters,
            'inertia': self.inertia_,
            'dtype': str(self.dtype),  # Save dtype info
        }
        
        if self.feature_importance_ is not None:
            model_data['feature_importance'] = self.feature_importance_.tolist()
            
        if self.cluster_feature_importance_ is not None:
            model_data['cluster_feature_importance'] = self.cluster_feature_importance_.tolist()
        
        if self.scaler is not None:
            scaler_data = {
                'mean': self.scaler.mean_.float().cpu().numpy().tolist() if self.scaler.mean_ is not None else None,
                'std': self.scaler.std_.float().cpu().numpy().tolist() if self.scaler.std_ is not None else None,
                'method': self.scaler.method,
                'device': self.scaler.device,
                'fitted': self.scaler.fitted,
                'dtype': str(self.scaler.dtype)
            }
            with open(os.path.join(model_dir, 'scaler.json'), 'w') as f:
                json.dump(scaler_data, f)
        
        with open(os.path.join(model_dir, 'kmeans_model.json'), 'w') as f:
            json.dump(model_data, f)
            
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir, device="cuda:0", dtype=None):
        """Load model with dtype support"""
        with open(os.path.join(model_dir, 'kmeans_model.json'), 'r') as f:
            model_data = json.load(f)
        
        # Get dtype from saved model or use provided
        if dtype is None:
            dtype_str = model_data.get('dtype', 'torch.float32')
            dtype = getattr(torch, dtype_str.split('.')[-1])
        
        n_clusters = model_data['n_clusters']
        model = cls(n_clusters=n_clusters, device=device, dtype=dtype)
        model.cluster_centers_ = torch.tensor(model_data['cluster_centers'], dtype=dtype, device=device)
        model.inertia_ = model_data['inertia']
        
        if 'feature_importance' in model_data:
            model.feature_importance_ = np.array(model_data['feature_importance'])
            
        if 'cluster_feature_importance' in model_data:
            model.cluster_feature_importance_ = np.array(model_data['cluster_feature_importance'])
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.json')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            scaler_dtype = getattr(torch, scaler_data.get('dtype', 'torch.float32').split('.')[-1])
            model.scaler = CustomScaler(device=device, method=scaler_data['method'], dtype=scaler_dtype)
            model.scaler.mean_ = torch.tensor(scaler_data['mean'], dtype=scaler_dtype, device=device) if scaler_data['mean'] is not None else None
            model.scaler.std_ = torch.tensor(scaler_data['std'], dtype=scaler_dtype, device=device) if scaler_data['std'] is not None else None
            model.scaler.fitted = scaler_data['fitted']
        else:
            model.scaler = None
        
        print(f"Model loaded from {model_dir} with dtype {dtype}")
        return model
        
    def visualize_feature_importance(self, feature_names=None, top_n=20):
        """
        Create visualization data for feature importance
        
        Args:
            feature_names: Optional list of feature names
            top_n: Number of top features to visualize
            
        Returns:
            Visualization data dictionary
        """
        if self.feature_importance_ is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
            
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance_))]
            
        # Get indices of top N important features
        top_indices = np.argsort(self.feature_importance_)[::-1][:top_n]
        
        # Get names and values
        top_names = [feature_names[i] for i in top_indices]
        top_values = [float(self.feature_importance_[i]) for i in top_indices]
        
        return {
            "feature_names": top_names,
            "importance_values": top_values
        }
    



