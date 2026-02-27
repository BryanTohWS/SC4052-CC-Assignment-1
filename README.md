# Assignment 1: Tutorial 1 Question 6

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone Repository
```bash
git clone https://github.com/BryanTohWS/SC4052-CC-Assignment-1.git
cd SC4052-CC-Assignment-1
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
## Running the Experiments

### Running Individual Experiments
```bash
# Experiment 1: Optimal Prediction Horizon
python experiment1_prediction_horizon.py

# Experiment 2: Fairness Under Model Divergence
python experiment2_fairness_divergence.py
```