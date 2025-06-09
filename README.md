# DAIR-FedMoE: Hierarchical MoE for Federated Encrypted Traffic Classification

This repository contains the implementation of DAIR-FedMoE, a novel framework for federated encrypted traffic classification that addresses distributed feature drift, concept drift, and label drift.

## Features

- Hierarchical Mixture of Experts (HMoE) layer with GShard Transformer backbone
- Local drift detection using Jensen-Shannon divergence
- Adaptive loss reweighting via expert confidence
- RL-based expert lifecycle management
- Differential privacy and secure aggregation
- Support for multiple encrypted traffic datasets:
  - ISCX-VPN
  - ISCX-Tor
  - CIC-IDS2017
  - UNSW-NB15

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dair-fedmoe.git
cd dair-fedmoe
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on a specific dataset:

```bash
python examples/train.py \
    --dataset iscx-vpn \
    --data_path /path/to/dataset \
    --num_clients 10 \
    --num_rounds 100 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_experts 8 \
    --hidden_dim 256 \
    --num_heads 8 \
    --dp_epsilon 1.0 \
    --dp_delta 1e-5 \
    --output_dir outputs
```

### Configuration

You can also provide a YAML configuration file:

```bash
python examples/train.py \
    --config configs/default.yaml \
    --dataset iscx-vpn \
    --data_path /path/to/dataset
```

Example configuration file (`configs/default.yaml`):
```yaml
model:
  num_experts: 8
  hidden_dim: 256
  num_heads: 8
  dropout: 0.1
  ff_expansion: 4
  num_layers: 6
  expert_dim: 512
  aux_loss_weight: 0.01
  drift_window_size: 100
  drift_threshold: 0.1
  reference_update_rate: 0.1

training:
  num_clients: 10
  num_rounds: 100
  learning_rate: 1e-3
  batch_size: 32
  ppo_clip_ratio: 0.2
  ppo_value_coef: 0.5
  ppo_entropy_coef: 0.01
```

## Project Structure

```
dair_fedmoe/
├── models/
│   ├── dair_fedmoe.py
│   ├── gshard_transformer.py
│   └── hierarchical_moe.py
├── drift/
│   └── drift_detector.py
├── rl/
│   └── expert_manager.py
├── training/
│   └── federated_trainer.py
├── privacy/
│   └── privacy_mechanism.py
├── utils/
│   └── dataset.py
├── config.py
└── __init__.py

examples/
├── train.py
└── evaluate.py

configs/
└── default.yaml

tests/
└── ...

docs/
└── ...
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{dair-fedmoe,
  title={DAIR-FedMoE: Hierarchical MoE for Federated Encrypted Traffic Classification under Distributed Feature, Concept, and Label Drift},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on ...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 