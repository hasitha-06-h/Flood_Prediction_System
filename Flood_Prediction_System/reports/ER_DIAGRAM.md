# Entity-Relationship Diagram

This is the ER diagram for the AquaGuard SQLite schema (see `database.py`).
GitHub, GitLab and most Markdown viewers render Mermaid diagrams natively -
open this file in the repo to see it rendered, or paste the block into
https://mermaid.live for a standalone view.

```mermaid
erDiagram
    USERS {
        int id PK
        string username
        string email
        string created_at
    }

    WEATHER_DATA {
        int id PK
        float annual_rainfall
        float cloud_cover
        float temperature
        float humidity
        float seasonal_rainfall
        string submitted_at
    }

    PREDICTION_HISTORY {
        int id PK
        int weather_data_id FK
        string model_used
        int prediction
        float probability
        string risk_level
        string predicted_at
    }

    ML_MODEL {
        string model_name PK
        float accuracy
        float precision
        float recall
        float f1_score
        int trained_on_rows
        string trained_at
    }

    WEATHER_DATA ||--o{ PREDICTION_HISTORY : "produces"
    ML_MODEL ||--o{ PREDICTION_HISTORY : "generates"
    USERS ||--o{ PREDICTION_HISTORY : "submits (future auth)"
```

### Notes on the schema

- `weather_data` stores exactly what a visitor typed into the Predict form -
  nothing is derived or scaled before it's persisted, so the raw input
  history stays reproducible.
- `prediction_history` is foreign-keyed to `weather_data` rather than
  duplicating the reading, keeping one row per prediction cycle.
- `ML_MODEL` isn't a physical SQLite table in the current build (it's
  represented by `models/model_metadata.json` instead) but is included
  here because the assignment brief calls for it as a conceptual entity -
  promoting it to a real table is a one-migration change if multiple
  model versions ever need to be tracked side by side.
- `users` exists as a placeholder for a future login flow; no route
  currently writes to it.
