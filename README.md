# Neural Network Pruning Strategies for Time-Series Forecasting

This repository contains a specialized research framework for **Hidden Layer Pruning** methodologies applied to Energy Consumption Forecasting. The core objective is to reduce the computational footprint of Deep Learning models while preserving their predictive power through innovative weight reconfiguration algorithms.

---

## 💻 Technical Environment

The project is optimized for the following stack:
* **Python:** 3.11.5
* **TensorFlow:** 2.17.0
* **Keras:** 3.11.3 (Leveraging the Keras 3 unified backend)
* **Hardware Recommendation:** Execution on GPU is supported but not mandatory due to the optimized pruning loops.

---

## 📂 Repository Structure

The project is structured into two specialized Jupyter Notebooks:

### 1. `1.Pruning_5HL_Model-(FullWorkflow).ipynb`
**The Complete Experimental Pipeline.**
* **Full Automation:** This notebook is a self-contained execution environment. It handles everything from data ingestion to final statistical reporting.
* **Workflow Requirements:** * **Data:** Requires a `dataset.csv` file (not included) containing time-series consumption data.
    * **Customization:** Users can modify the base architecture (default is a 5-Hidden Layer Dense network).
    * **Selection:** Includes a configuration cell to choose between the three implemented pruning methodologies.
* **Iterative Loop:** Automatically prunes each layer, re-trains the model (fine-tuning), and stores results for comparative analysis.

### 2. `2.Pruning_Strategies.ipynb`
**The Modular Algorithm Library.**
* **Pure Logic:** Contains the standalone Python functions for each pruning strategy.
* **Ideal for Research:** Use this notebook to inspect how weights are mathematically mapped and transformed during the pruning process without the overhead of the full training cycle.

---

## 🛠️ Pruning Methodologies Explained

Traditional pruning often results in "broken" networks. This project solves this by using weight redistribution:

### A. Backward Strategy (BS)

This strategy reconfigures the network by looking "backward" from the layer targeted for removal. It uses **absolute mean similarity** to map the importance of neurons in the target layer back to the preceding layer. This ensures that the features learned by the pruned layer are "absorbed" by its predecessor.

### B. Forward Strategy (FS)
The Forward Strategy focuses on the interface between the pruned layer and the subsequent one. It adjusts the input weight matrix of the "next" layer to accommodate the new input dimensions coming from the layer before the pruned one.

### C. Standard Pruning (Classic)
The "Control Group" strategy. It removes the layer and re-initializes the adjacent connections. This is used as a benchmark to demonstrate the superior performance of the BS and FS algorithms.

---

## 📊 Data Processing & Evaluation Metrics

To ensure scientific validity, the project follows a rigorous evaluation protocol:

### 1. Data Normalization & Denormalization
Models are trained on normalized data ($[0, 1]$ range) to speed up convergence. However, all performance metrics are calculated after **denormalizing** predictions back to the original scale (e.g., actual electricity consumption in kWh), providing real-world context.

### 2. Global MAPE & Error Stability
* **MAPE (Mean Absolute Percentage Error):** The primary accuracy indicator.
* **Standard Deviation of MAPE:** Unlike standard models, this framework calculates the standard deviation of absolute percentage errors across all test samples. This measures the **reliability** of the pruning—a low standard deviation means the model's accuracy is consistent and doesn't vary wildly between different prediction windows.

### 3. Fine-Tuning Performance
The project monitors the efficiency of the "recovery" phase:
* **Epoch Count:** The number of iterations required to reach the best validation loss after pruning.
* **Training Time:** Measured in seconds to quantify the speed of each pruning strategy.

---

## 🚀 Getting Started

### Installation
Ensure your environment matches the required versions to avoid API conflicts:
```bash
pip install tensorflow==2.17.0 keras==3.11.3 pandas numpy scikit-learn matplotlib
