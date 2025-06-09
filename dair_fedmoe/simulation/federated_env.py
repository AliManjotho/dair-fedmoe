import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader, Subset, random_split
from ..models.dair_fedmoe import DAIRFedMoE
from ..utils.dataset import DatasetProcessor
from ..drift.drift_detector import DriftDetector
from ..privacy.privacy_mechanism import PrivacyMechanism

class FederatedEnvironment:
    def __init__(
        self,
        config: dict,
        dataset_processor: DatasetProcessor,
        num_clients: int = 10,
        drift_type: str = "feature",  # "feature", "concept", "label"
        drift_strength: float = 0.5,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5
    ):
        self.config = config
        self.dataset_processor = dataset_processor
        self.num_clients = num_clients
        self.drift_type = drift_type
        self.drift_strength = drift_strength
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        
        # Initialize components
        self.server_model = None
        self.client_models = []
        self.client_loaders = []
        self.drift_detectors = []
        self.privacy_mechanism = PrivacyMechanism(epsilon=privacy_epsilon, delta=privacy_delta)
        
    def setup(self, dataset_path: str):
        """Setup the federated learning environment"""
        # Load and preprocess dataset
        dataset = self.dataset_processor.load_dataset(dataset_path)
        
        # Split dataset among clients
        self._split_dataset(dataset)
        
        # Initialize server model
        self.server_model = DAIRFedMoE(self.config)
        
        # Initialize client models and drift detectors
        for _ in range(self.num_clients):
            client_model = DAIRFedMoE(self.config)
            client_model.load_state_dict(self.server_model.state_dict())
            self.client_models.append(client_model)
            
            drift_detector = DriftDetector(
                window_size=self.config["drift"]["window_size"],
                threshold=self.config["drift"]["threshold"]
            )
            self.drift_detectors.append(drift_detector)
    
    def _split_dataset(self, dataset: torch.utils.data.Dataset):
        """Split dataset among clients with non-IID distribution"""
        # Calculate sizes for each client
        total_size = len(dataset)
        client_sizes = self._generate_non_iid_sizes(total_size)
        
        # Split dataset
        splits = random_split(dataset, client_sizes)
        
        # Create data loaders for each client
        for split in splits:
            loader = DataLoader(
                split,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True
            )
            self.client_loaders.append(loader)
    
    def _generate_non_iid_sizes(self, total_size: int) -> List[int]:
        """Generate non-IID dataset sizes for clients"""
        # Use Dirichlet distribution to create non-IID splits
        alpha = 0.5  # Controls non-IIDness (lower = more non-IID)
        proportions = np.random.dirichlet([alpha] * self.num_clients)
        sizes = [int(p * total_size) for p in proportions]
        
        # Adjust for rounding errors
        sizes[-1] = total_size - sum(sizes[:-1])
        return sizes
    
    def inject_drift(self, client_idx: int, round_num: int):
        """Inject drift into client's data"""
        if self.drift_type == "feature":
            self._inject_feature_drift(client_idx)
        elif self.drift_type == "concept":
            self._inject_concept_drift(client_idx)
        elif self.drift_type == "label":
            self._inject_label_drift(client_idx)
    
    def _inject_feature_drift(self, client_idx: int):
        """Inject feature drift by modifying input features"""
        loader = self.client_loaders[client_idx]
        for batch in loader:
            features, _ = batch
            # Add noise to features
            noise = torch.randn_like(features) * self.drift_strength
            features.add_(noise)
    
    def _inject_concept_drift(self, client_idx: int):
        """Inject concept drift by modifying feature-label relationships"""
        loader = self.client_loaders[client_idx]
        for batch in loader:
            features, labels = batch
            # Modify labels based on feature patterns
            mask = (features[:, 0] > 0.5)  # Example condition
            labels[mask] = 1 - labels[mask]  # Flip labels
    
    def _inject_label_drift(self, client_idx: int):
        """Inject label drift by modifying label distribution"""
        loader = self.client_loaders[client_idx]
        for batch in loader:
            _, labels = batch
            # Randomly flip some labels
            mask = torch.rand_like(labels) < self.drift_strength
            labels[mask] = 1 - labels[mask]
    
    def train_round(self, round_num: int) -> Dict[str, float]:
        """Execute one round of federated training"""
        client_updates = []
        client_metrics = []
        
        # Train each client
        for client_idx in range(self.num_clients):
            # Inject drift if needed
            if round_num > 0 and round_num % 10 == 0:  # Inject drift every 10 rounds
                self.inject_drift(client_idx, round_num)
            
            # Train client
            client_model = self.client_models[client_idx]
            client_loader = self.client_loaders[client_idx]
            drift_detector = self.drift_detectors[client_idx]
            
            # Train and get updates
            update, metrics = self._train_client(
                client_model,
                client_loader,
                drift_detector,
                round_num
            )
            
            # Apply privacy mechanism
            private_update = self.privacy_mechanism.apply(update)
            client_updates.append(private_update)
            client_metrics.append(metrics)
        
        # Aggregate updates
        self._aggregate_updates(client_updates)
        
        # Calculate average metrics
        avg_metrics = self._average_metrics(client_metrics)
        return avg_metrics
    
    def _train_client(
        self,
        model: DAIRFedMoE,
        loader: DataLoader,
        drift_detector: DriftDetector,
        round_num: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train a single client"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["training"]["learning_rate"])
        metrics = {"loss": 0.0, "accuracy": 0.0}
        
        for batch in loader:
            features, labels = batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = model.compute_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            metrics["loss"] += loss.item()
            metrics["accuracy"] += (outputs.argmax(dim=1) == labels).float().mean().item()
        
        # Normalize metrics
        metrics["loss"] /= len(loader)
        metrics["accuracy"] /= len(loader)
        
        # Detect drift
        drift_detector.update(metrics)
        if drift_detector.detect_drift():
            print(f"Drift detected in client at round {round_num}")
        
        # Get model update
        update = {
            name: param.data.clone() - self.server_model.state_dict()[name]
            for name, param in model.named_parameters()
        }
        
        return update, metrics
    
    def _aggregate_updates(self, updates: List[Dict[str, torch.Tensor]]):
        """Aggregate client updates using FedAvg"""
        aggregated = {}
        for name in self.server_model.state_dict().keys():
            aggregated[name] = torch.stack([update[name] for update in updates]).mean(0)
        
        # Update server model
        server_state = self.server_model.state_dict()
        for name, param in server_state.items():
            param.add_(aggregated[name])
        self.server_model.load_state_dict(server_state)
        
        # Update client models
        for client_model in self.client_models:
            client_model.load_state_dict(server_state)
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics across clients"""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return avg_metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the server model on test data"""
        self.server_model.eval()
        metrics = {"loss": 0.0, "accuracy": 0.0}
        
        with torch.no_grad():
            for batch in test_loader:
                features, labels = batch
                outputs = self.server_model(features)
                loss = self.server_model.compute_loss(outputs, labels)
                
                metrics["loss"] += loss.item()
                metrics["accuracy"] += (outputs.argmax(dim=1) == labels).float().mean().item()
        
        # Normalize metrics
        metrics["loss"] /= len(test_loader)
        metrics["accuracy"] /= len(test_loader)
        
        return metrics 