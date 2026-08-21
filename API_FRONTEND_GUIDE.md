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
| Cargo transport assessment | `POST /transport/cargo-assessment` | JSON with product name, approved temperature limits, optional humidity limits, and optional sensor values/device. |
| Create cargo profile | `POST /transport/profiles` | Save product limits once and receive a `profile_id`. |
| List cargo profiles | `GET /transport/profiles` | List profiles for the current API service; optional `customer_id` filter. |
| Get cargo profile | `GET /transport/profiles/{profile_id}` | Read one saved cargo profile. |

## Cargo Transport

The API does not hardcode storage requirements for apples, oranges, or other products. The carrier supplies the approved range from its transport specification, customer contract, or food-safety specialist.

Example request:

```json
{
	"product_name": "mere",
	"min_temperature": 2,
	"max_temperature": 8,
	"min_humidity": 80,
	"max_humidity": 95,
	"device_identifier": "truck-01-sensor",
	"temperature": 9.5,
	"humidity": 88
}
```

If `temperature` or `humidity` is omitted, the API reads the latest values for `device_identifier` from Supabase. The response contains `status`, `alerts`, `recommended_temperature`, and `recommended_action`. The API evaluates and recommends a setpoint; an external controller/PLC must perform the physical refrigeration adjustment.

For additional parameters, send per-parameter limits and values:

```json
{
	"product_name": "portocale",
	"min_temperature": 3,
	"max_temperature": 8,
	"parameter_limits": {
		"temperature": {"min_value": 3, "max_value": 8},
		"humidity": {"min_value": 85, "max_value": 95},
		"co2": {"max_value": 1000},
		"voc": {"max_value": 250}
	},
	"parameter_values": {
		"temperature": 6,
		"humidity": 90,
		"co2": 1200,
		"voc": 180
	}
}
```

Supported parameter names are `temperature`, `humidity`, `pm25`, `pm10`, `co2`, and `voc`. The response exposes `parameter_status` and `parameter_values`, so a client can decide whether to adjust cooling, ventilation, or another actuator.

### Reusable Profile

Create the product profile once:

```json
{
	"profile_id": "firma-1-mere-standard",
	"customer_id": "firma-1",
	"product_name": "mere",
	"min_temperature": 2,
	"max_temperature": 8,
	"min_humidity": 80,
	"max_humidity": 95,
	"parameter_limits": {
		"co2": {"max_value": 1000}
	}
}
```

Then assess each truck using only the saved profile and current readings:

```json
{
	"profile_id": "firma-1-mere-standard",
	"device_identifier": "truck-01-sensor",
	"parameter_values": {
		"temperature": 9,
		"humidity": 88,
		"co2": 1200
	}
}
```

Profiles are held in the API process memory. For durable multi-instance production storage, persist the same profile model in a customer-scoped Supabase table and enforce authentication/authorization by `customer_id`.

## Android Screens

Create the following screens or actions:

1. Dashboard: current prediction and anomaly check.
2. Forecast: horizon selector and results from the `forecast` array.
3. Manual prediction: numeric inputs for the five required sensor fields.
4. Training: model selector and aggregation interval.
5. Settings: API status from `/health` and data-source status from `/health/data`.

Responses use JSON. Show `detail` when an HTTP response is not successful. A `400` from prediction, forecast, anomaly, or training normally means the configured Supabase data is missing or insufficient.

For Random Forest training, use `training_report.technical_details.evolution` to render the OOB score by iteration. Display `training_report.model_info.n_estimators` as the configured number of trees. Other algorithms can return different evolution metrics or no evolution data.