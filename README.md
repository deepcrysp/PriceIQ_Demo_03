
# PriceIQ 2.0

AI-powered pricing intelligence proof of concept using SAP pricing agreement data.

## Features

- Executive pricing agreement overview
- SAP agreement explorer
- Machine-learning pricing anomaly detection
- Material and unit price benchmarking
- Agreement expiry and renewal monitoring
- Customer agreement profiles
- Hypothetical margin guardrail sandbox
- Data readiness assessment

## AI and analytics

PriceIQ uses local Python machine learning, including Isolation Forest, along with statistical and rule-based checks.

No external AI API key is required.

Anomaly flags are review indicators, not confirmed pricing violations.

## Data

Place the normalized customer agreement CSV here:

data/customer_agreements.csv

The application also searches nested folders for this file.

## Run locally

Install dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to app.py.
5. Deploy.

## Data limitations

The current SAP agreement extracts do not contain invoice quantities,
realized selling prices, product costs, rebates, freight or competitor
pricing.

Therefore, this version does not claim to calculate validated price
elasticity, actual customer profitability or optimized selling prices.

These capabilities require additional source data.
