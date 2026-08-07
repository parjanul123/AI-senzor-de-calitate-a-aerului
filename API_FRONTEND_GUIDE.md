# Frontend And Android API Guide

Base URL:

```text
https://ai-senzor-de-calitate-a-aerului-production.up.railway.app
```

The live interactive API specification is available at `/docs`. Android clients can import the OpenAPI contract from `/openapi.json`.

| UI control | Method and path | Request |
|---|---|---|
| Current prediction | `POST /predict` | No body. Uses the latest Supabase measurement. |
| Forecast | `POST /predict?include_forecast=true&forecast_horizons=1,3,6,12,24` | No body. `forecast` in the response contains one result for each horizon. |
| Manual prediction | `POST /predict-custom` | JSON: `temperature`, `humidity`, `pm25`, `pm10`, `co2`. |
| Anomaly check | `POST /anomaly` | No body. Uses the latest Supabase measurement. |
| Model training | `POST /train` | JSON: `training_model`, `aggregation_hours`, optional `aggregation_minutes`. |
| API status | `GET /health` | No body. |
| Sensor-data status | `GET /health/data` | No body. Verifies that the Railway service can read Supabase measurements. |

## Android Screens

Create the following screens or actions:

1. Dashboard: current prediction and anomaly check.
2. Forecast: horizon selector and results from the `forecast` array.
3. Manual prediction: numeric inputs for the five required sensor fields.
4. Training: model selector and aggregation interval.
5. Settings: API status from `/health` and data-source status from `/health/data`.

Responses use JSON. Show `detail` when an HTTP response is not successful. A `400` from prediction, forecast, anomaly, or training normally means the configured Supabase data is missing or insufficient.