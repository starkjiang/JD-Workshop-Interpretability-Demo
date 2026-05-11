# 🚜 AI Interpretability Demo — John Deere Workshop

A Streamlit app demonstrating **explainable AI for equipment failure prediction**,
built around the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

---

## What the Demo Shows

| Part | Title | What the Audience Sees |
|------|-------|------------------------|
| **1** | Black Box Prediction | Live failure probability from slider inputs |
| **2** | Local Explanation | SHAP waterfall + bar chart for *this* prediction |
| **3** | Global Explanation | Model-wide feature importance + beeswarm plot |
| **4** | Counterfactual Reasoning | "What-if" maintenance intervention simulator |

---

## Quickstart

### 1. Clone / download the repo

```bash
git clone https://github.com/starkjiang/JD-Workshop-Interpretability-Demo.git
cd jd_interpretability_demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ recommended.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Repository Structure

```
jd_interpretability_demo/
├── app.py               # Main Streamlit application
├── model.py             # Data loading, model training, SHAP explainer
├── ai4i2020.csv         # AI4I 2020 dataset (10 000 rows)
├── requirements.txt     # Python dependencies
├── john_deere_logo.png  # John Deere Logo
└── README.md
```

---

## Dataset

**AI4I 2020 Predictive Maintenance** — 10 000 synthetic machine records with:

| Feature | Description |
|---------|-------------|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine process temperature |
| Rotational speed [rpm] | Spindle rotation speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear time |
| **Machine failure** | Binary target (1 = failure) |

Class balance: ~3.4% failure rate (339 / 10 000).

---

## Model

- **Algorithm:** `RandomForestClassifier` (scikit-learn)
- **Class weighting:** `balanced` (handles imbalance)
- **Depth:** max 8 levels
- **Explainer:** `shap.TreeExplainer`

The model trains automatically on first launch and is cached for the session.

---

## Demo Tips for Presenters

1. **Start with sliders at default** — show the baseline prediction.
2. **Click "🔴 High Risk" preset** to jump to a dramatic failure scenario.
3. **Walk through Parts 1→4** in order — each section builds on the last.
4. **Counterfactual section** is the crowd-pleaser: show that reducing tool wear
   from 220 → 70 min (replacing the tool) cuts risk dramatically.
5. Ask the audience: *"Would you trust this model's maintenance recommendation
   more or less now that you can see why it made the prediction?"*
