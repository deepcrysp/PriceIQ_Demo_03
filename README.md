# PriceIQ 2.0 | Pricing Intelligence & Margin Governance

PriceIQ is an ERP-independent pricing intelligence proof of concept built around SAP pricing agreement extracts. It is intended for exploration and decision support, not as a production pricing engine or a replacement for SAP/Pricefx.

## What is included

- Executive agreement overview
- Agreement explorer with region and district filters
- Statistical and Isolation Forest anomaly investigation signals
- Descriptive material/unit agreement-value benchmarking
- Expiry and renewal monitoring
- Customer agreement profiles
- Hypothetical margin guardrail sandbox using user-entered cost, list price and policy assumptions
- Data readiness assessment
- Multi-file CSV and Excel upload (CSV, XLSX and XLS)
- Uploaded Data Preview with source-file summary, normalized rows, column coverage and CSV download

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Upload your own agreement data

1. Start PriceIQ.
2. In the left sidebar, use **Upload agreement data (CSV or Excel)**.
3. Select one or more CSV, XLSX or XLS extracts.
4. The app normalizes common SAP/source aliases and combines the readable files.
5. Open **Uploaded Data Preview** to inspect rows, file-level counts and column coverage.

If no files are uploaded, PriceIQ loads `data/customer_agreements.csv` from the repository. Uploaded files are held in the Streamlit session and are not written back to GitHub or persisted by the app. For large files, upload one extract at a time if your hosting provider has request-size limits.

## Expected fields and common SAP aliases

The application maps common source names to these normalized fields:

| Normalized field | Common source aliases |
|---|---|
| customer | customer_id, customer_number, sold_to, KUNNR |
| material | material_id, material_number, MATNR |
| unit | UOM, sales_unit, VRKME, MEINS |
| agreement_value | agreement_price, condition_value, KBETR, net_price |
| valid_from | start_date, effective_from, DATAB |
| valid_to | end_date, effective_to, DATBI |
| region | sales_region |
| district | sales_district |
| agreement_type | condition_type, KSCHL |
| agreement_id | agreement_number, condition_record, KNUMH |
| currency | currency_code, WAERS |
| description | material_description, MAKTX |

Missing optional fields are populated with neutral defaults for the app. Keep original source extracts unchanged and validate field mappings before using outputs in business decisions. A blank currency field is not interpreted as USD or any other currency.

A starter CSV header and sample rows are available at `data/upload_template.csv`.

## Deploy on Streamlit Community Cloud

1. Create or select a GitHub repository.
2. Push the project files.
3. In Streamlit Community Cloud, select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

Keep customer data in a **private repository** or upload it through the app at runtime. Do not publish confidential customer SAP extracts in a public repository. The bundled `data/customer_agreements.csv` is the source dataset currently used by this customer-specific POC. Check your organization's data-handling approval before distributing it.

## Important interpretation and data limitations

The bundled source is SAP agreement data, not invoice-level transaction history. Agreement values are SAP-maintained agreement/condition values and are not necessarily realized transaction prices.

The supplied extracts do not provide invoice quantities, realized selling prices, product costs, rebates, freight, cost-to-serve, competitor prices or quote history. Consequently:

- Material/unit comparisons are descriptive agreement-value benchmarks, not market price benchmarks.
- Anomaly flags are investigation signals, not confirmed pricing violations.
- The app does not calculate validated price elasticity, demand response, actual gross/pocket margin, realized revenue forecasts or optimized selling prices.
- Margin guardrail scenarios are hypothetical. Cost, list price, minimum margin and maximum discount are entered by the user and are not inferred from the agreement extract.

Use the Data Readiness page to understand source coverage and identify data needed for more advanced pricing analytics.

## Technical notes

- Streamlit UI and pandas data preparation
- scikit-learn Isolation Forest for statistical investigation signals
- No external AI API key required
- Uploaded data is session-scoped; it is not stored as a server-side customer data repository
