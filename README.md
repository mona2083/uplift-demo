# 🎯 Promo ROI Predictor

> Stop sending coupons to everyone. Use AI to find the exact customers where a promotion will actually change their behavior.

A Streamlit app that uses Uplift Modeling (T-Learner) to classify customers into four segments and calculate the ROI difference between blanket coupon distribution vs. precision targeting. 

---

## Live Demo

🔗 [Open App](https://uplift-demo-2hybmyogzljzqureo9qvru.streamlit.app/) 

---

## The Problem It Solves

Most retailers send promotional offers to their entire customer base — a 20% OFF email to 100,000 members. This is wasteful because:

- **Sure Things** buy anyway — the discount just reduces margin
- **Sleeping Dogs** are loyal customers who feel pestered and may churn
- **Lost Causes** won't buy regardless of the discount
- Only **Persuadables** — typically ~20% of customers — actually change their behavior because of the promotion

This app identifies exactly who is a Persuadable and shows the financial impact of targeting only them.

---

## The 4 Customer Segments

| Segment | Purchase Probability | Coupon Effect | Action |
|---|---|---|---|
| 💎 Sure Things (鉄板層) | High | Minimal | Don't waste coupons |
| 🎯 Persuadables (説得可能層) | Low | High uplift | Target all budget here |
| 🚪 Lost Causes (無関心層) | Low | No effect | Save the budget |
| ⚠️ Sleeping Dogs (あまのじゃく層) | High | Negative | Never send promotions |

---

## Features & Technical Highlights

### 🧠 Robust Causal Inference
- **T-Learner (Meta-Learner)**: Uses dual Gradient Boosting Classifiers optimized with subsampling and depth constraints to prevent overfitting on imbalanced treatment/control splits.
- **Score Stabilization**: Outputs are mathematically clipped to valid probability boundaries to ensure reliable ROI calculations.

### 🔍 Explainable AI (XAI)
- **Feature Importance Heatmap**: Uses One-vs-Rest Random Forest classifiers to interpret *why* a customer belongs to a certain segment.
- **Persona Generation**: Translates raw data into distinct customer personas with key behavioral traits.

### 💰 ROI Simulation
- Side-by-side metrics: send to all customers vs. Persuadables only.
- Compares coupon cost, incremental revenue (gross profit), and net profit dynamically.

### 📊 Modular UI Architecture
- Clean separation of concerns: Orchestration (`app.py`), ML Logic (`model.py`), and complex Plotly rendering (`visuals.py`).
- Interactive 4-quadrant scatter plot with adjustable thresholds.
- Full English / 日本語 toggle.

---

## Tech Stack

| Component | Technology |
|---|---|
| Uplift Model | T-Learner (Two-Model Approach) with `scikit-learn` GradientBoostingClassifier |
| Feature Importance | One-vs-Rest `RandomForestClassifier` |
| UI | `streamlit` |
| Visualization | `plotly` |
| Data | Synthetic generator (NumPy) + real CSV upload |

---

## How the Model Works

The **T-Learner** trains two separate Gradient Boosting models:

- **Model T** — trained only on customers who received a coupon
- **Model C** — trained only on customers who did not

For each customer, both models predict purchase probability. The **uplift score** is the difference:

```text
Uplift = P(buy | coupon) - P(buy | no coupon)
```

A score of +0.25 means the coupon increases purchase probability by 25 percentage points for that customer. A score of -0.15 means the coupon would make them 15% less likely to buy (Sleeping Dog).

---

## Project Structure

```text
promo-roi-predictor/
├── app.py              # Streamlit UI & state orchestration
├── model.py            # T-Learner logic, score clipping, ROI calculation
├── visuals.py          # Separated Plotly charting and XAI rendering logic
├── constants.py        # Styling, bilingual labels, and persona definitions
├── lang.py             # i18n dictionary (EN/JA)
├── data_generator.py   # Synthetic customer data generator
├── requirements.txt
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/mona2083/promo-roi-predictor.git
cd promo-roi-predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Demo Data
1. Select **"Use demo data"** in the sidebar
2. Set the number of customers (1,000–20,000) and random seed
3. Click **"🚀 Run Model"** to train and predict

### Your Own Data
Upload a CSV with the following format:

```csv
customer_id,treatment,conversion,age,past_spend,visit_freq,days_since_last,avg_basket
1,1,1,35,250,8,12,65
2,0,0,52,80,2,90,30
```

| Column | Type | Description |
|---|---|---|
| `customer_id` | integer | Unique customer ID |
| `treatment` | 0 or 1 | Whether coupon was sent (1=yes, 0=no) |
| `conversion` | 0 or 1 | Whether customer purchased (1=yes, 0=no) |
| feature columns | numeric | Age, spend, frequency, etc. (any number of columns) |

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

To deploy your own instance:
1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file

---

## Author

**Manami Oyama** — AI Engineer / Product Manager  
🌺 Honolulu, Hawaii  
🔗 [Portfolio](https://mona2083.github.io/portfolio-2026/index.html) | [GitHub](https://github.com/mona2083) | [LinkedIn](https://www.linkedin.com/in/manami-oyama/)