# Quantum-ML-IDS-CICIDS2017
Quantum Machine Learning per Network Intrusion Detection: benchmark VQC/QSVC su CICIDS2017  con analisi comparativa di performance, scalabilità e costo computazionale rispetto a baseline classici.

## Objective

Evaluate the applicability of QML algorithms to real-world cybersecurity problems, identifying sweet-spot configurations 
between performance metrics (F1-Score, Precision, Recall) and computational cost.

## Models Tested

- **VQC efficient_su2** (4-12 qubits, multiple ansatz repetitions)
- **VQC real_amplitudes** (4-10 qubits)
- **VQC Re-Upload** (data re-uploading circuit)
- **QSVC** (Quantum Support Vector Classifier, 6-12 qubits)
- **Random Forest** (classical baseline, 6-20 PCA features)

## Key Results

- **Sweet spot identified**: VQC efficient_su2 with 6 qubits, sample size 2000
- **Best F1-Score**: 0.9XX (VQC) vs 0.9XX (Random Forest baseline)
- **Trade-off**: training times 10-100x higher than RF for marginal improvements
- **Scalability**: performance degrades beyond 10 qubits due to sampling limitations

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Quantum-ML-detection-cicids2017.git
cd Quantum-ML-detection-cicids2017

# Install dependencies
pip install -r requirements.txt

# Download CICIDS2017 dataset
# (instructions in data/README.md)

# Generate plots
python scripts/genera_grafici.py

# Run experiments
jupyter notebook notebooks/RF_VQC_QSVC_CICIDS2017.ipynb
```

## Dataset

CICIDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System Dataset)
- Source: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
- Preprocessed with PCA for dimensionality reduction (6-20 features)
- Binary classification: Benign vs Attack

**Setup**: After downloading, place the dataset files in a `.archive/` directory at the root of the repository. 
This folder is excluded from version control via `.gitignore`.

## Dependencies

- Python 3.8+
- Qiskit 0.39+
- Qiskit Machine Learning 0.5+
- scikit-learn 1.0+
- pandas, numpy, matplotlib
- openpyxl (for Excel reading)

See `requirements.txt` for complete list.

## Methodology

All quantum circuits executed on Qiskit's QASM simulator with 1024 shots.
Training performed with COBYLA optimizer (max iterations: 100-200).
Cross-validation and hyperparameter tuning details in the notebook.

## Results Visualization

The `scripts/genera_grafici.py` script generates 9+ publication-ready plots:
- F1-Score vs number of qubits
- Scaling analysis (F1 vs sample size)
- Quality vs training time trade-off
- Precision/Recall analysis for false positive/negative detection
- QSVC error analysis
- Configuration comparison (best model per type)
