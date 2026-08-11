import pandas as pd

from app.models import train_model as train_model_module
from app.services import anomaly_detector as anomaly_detector_module
from app.services import predictor as predictor_module
from app.services.anomaly_detector import detect_anomaly
from app.models.train_model import (
    load_model,
    train_and_save_isolation_forest,
    train_and_save_model,
    train_and_save_svm,
)
from app.models.xgboost_model import (
    XGBOOST_FEATURE_COLUMNS_STANDARD,
    XGBOOST_FEATURE_COLUMNS_EXTENDED,
    XGBoostModel,
    train_and_save_xgboost,
)
from app.api.main import CustomPredictionRequest


def test_custom_prediction_request_has_no_pm_bounds():
    request = CustomPredictionRequest(
        temperature=20.0,
        humidity=50.0,
        pm25=-1.0,
        pm10=10_000.0,
        co2=800.0,
    )

    assert request.pm25 == -1.0
    assert request.pm10 == 10_000.0


def _sample_measurements_dataframe():
    return pd.DataFrame(
        [
            {
                "temperatura": 24.0,
                "umiditate": 45.0,
                "pm25": 12.0,
                "pm10": 20.0,
                "co2": 550.0,
                "quality_label": "good",
                "quality_label_source": "manual",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "temperatura": 31.0,
                "umiditate": 62.0,
                "pm25": 80.0,
                "pm10": 110.0,
                "co2": 1400.0,
                "quality_label": "poor",
                "quality_label_source": "manual",
                "created_at": "2026-01-02T00:00:00Z",
            },
            {
                "temperatura": 25.0,
                "umiditate": 50.0,
                "pm25": 28.0,
                "pm10": 45.0,
                "co2": 800.0,
                "quality_label": "moderate",
                "quality_label_source": "manual",
                "created_at": "2026-01-03T00:00:00Z",
            },
            {
                "temperatura": 26.0,
                "umiditate": 52.0,
                "pm25": 15.0,
                "pm10": 25.0,
                "co2": 650.0,
                "quality_label": "good",
                "quality_label_source": "expert_review",
                "created_at": "2026-01-04T00:00:00Z",
            },
            {
                "temperatura": 30.0,
                "umiditate": 60.0,
                "pm25": 55.0,
                "pm10": 85.0,
                "co2": 1200.0,
                "quality_label": "poor",
                "quality_label_source": "lab_reference",
                "created_at": "2026-01-05T00:00:00Z",
            },
            {
                "temperatura": 23.0,
                "umiditate": 40.0,
                "pm25": 10.0,
                "pm10": 18.0,
                "co2": 500.0,
                "quality_label": "good",
                "quality_label_source": "external_aqi_standard",
                "created_at": "2026-01-06T00:00:00Z",
            },
            {
                "temperatura": 29.0,
                "umiditate": 58.0,
                "pm25": 42.0,
                "pm10": 70.0,
                "co2": 980.0,
                "quality_label": "moderate",
                "quality_label_source": "manual",
                "created_at": "2026-01-07T00:00:00Z",
            },
            {
                "temperatura": 27.0,
                "umiditate": 55.0,
                "pm25": 35.0,
                "pm10": 60.0,
                "co2": 900.0,
                "quality_label": "moderate",
                "quality_label_source": "independent_sensor_fusion",
                "created_at": "2026-01-08T00:00:00Z",
            },
            {
                "temperatura": 22.0,
                "umiditate": 42.0,
                "pm25": 9.0,
                "pm10": 15.0,
                "co2": 480.0,
                "quality_label": "good",
                "quality_label_source": "manual",
                "created_at": "2026-01-09T00:00:00Z",
            },
            {
                "temperatura": 32.0,
                "umiditate": 65.0,
                "pm25": 70.0,
                "pm10": 120.0,
                "co2": 1500.0,
                "quality_label": "poor",
                "quality_label_source": "expert_review",
                "created_at": "2026-01-10T00:00:00Z",
            },
            {
                "temperatura": 24.0,
                "umiditate": 48.0,
                "pm25": 22.0,
                "pm10": 38.0,
                "co2": 720.0,
                "quality_label": "moderate",
                "quality_label_source": "manual",
                "created_at": "2026-01-11T00:00:00Z",
            },
        ]
    )


def test_train_and_save_model(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_model.pkl"
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    model = train_and_save_model(model_path=model_path)

    assert model_path.exists()
    assert model is not None

    loaded_model = load_model(model_path)
    prediction = loaded_model.predict([[25.0, 45.0, 20.0, 35.0, 600.0]])

    assert len(prediction) == 1


def test_train_and_save_svm(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_svm.pkl"
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    model = train_and_save_svm(model_path=model_path)

    assert model_path.exists()
    assert model is not None

    loaded_model = load_model(model_path)
    prediction = loaded_model.predict([[25.0, 45.0, 20.0, 35.0, 600.0]])

    assert len(prediction) == 1


def test_train_and_save_isolation_forest(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_if.pkl"
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    model = train_and_save_isolation_forest(model_path=model_path)

    assert model_path.exists()
    assert model is not None

    loaded_model = load_model(model_path)
    prediction = loaded_model.predict([[25.0, 45.0, 20.0, 35.0, 600.0]])

    assert len(prediction) == 1


def test_train_and_save_xgboost(tmp_path, monkeypatch):
    model_path = tmp_path / "xgboost.pkl"
    database_rows = _sample_measurements_dataframe()
    for index, feature_name in enumerate(["pressure", "gas", "lux", "pm1"]):
        database_rows[feature_name] = 1000.0 + index + database_rows.index

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    model = train_and_save_xgboost(model_path=model_path)

    assert model_path.exists()
    assert isinstance(model, XGBoostModel)

    loaded_model = load_model(model_path)
    sample = pd.DataFrame(
        [[25.0, 45.0, 1000.0, 1001.0, 1002.0, 600.0, 1003.0, 20.0, 35.0]],
        columns=XGBOOST_FEATURE_COLUMNS_EXTENDED,
    )
    assert len(loaded_model.predict(sample)) == 1
    assert loaded_model.predict_proba(sample).shape[0] == 1


def test_load_training_data_uses_database_measurements(monkeypatch):
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    X, y = train_model_module.load_training_data()

    assert len(X) == len(database_rows)
    assert set(train_model_module.FEATURE_COLUMNS).issubset(set(X.columns))
    assert set(y.unique()).issubset({"good", "moderate", "poor"})
    assert len(y) == len(X)


def test_supervised_training_requires_quality_label_source(monkeypatch):
    database_rows = pd.DataFrame(
        [
            {
                "temperatura": 22.0,
                "umiditate": 45.0,
                "pm25": 20.0,
                "pm10": 35.0,
                "co2": 650.0,
                "quality_label": "good",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "temperatura": 22.0,
                "umiditate": 45.0,
                "pm25": 20.0,
                "pm10": 110.0,
                "co2": 1500.0,
                "quality_label": "poor",
                "created_at": "2026-01-02T00:00:00Z",
            },
            {
                "temperatura": 23.0,
                "umiditate": 50.0,
                "pm25": 30.0,
                "pm10": 55.0,
                "co2": 900.0,
                "quality_label": "moderate",
                "created_at": "2026-01-03T00:00:00Z",
            },
            {
                "temperatura": 24.0,
                "umiditate": 52.0,
                "pm25": 25.0,
                "pm10": 45.0,
                "co2": 850.0,
                "quality_label": "moderate",
                "created_at": "2026-01-04T00:00:00Z",
            },
            {
                "temperatura": 25.0,
                "umiditate": 49.0,
                "pm25": 18.0,
                "pm10": 30.0,
                "co2": 700.0,
                "quality_label": "good",
                "created_at": "2026-01-05T00:00:00Z",
            },
            {
                "temperatura": 28.0,
                "umiditate": 57.0,
                "pm25": 48.0,
                "pm10": 78.0,
                "co2": 1100.0,
                "quality_label": "poor",
                "created_at": "2026-01-06T00:00:00Z",
            },
            {
                "temperatura": 21.0,
                "umiditate": 40.0,
                "pm25": 12.0,
                "pm10": 22.0,
                "co2": 560.0,
                "quality_label": "good",
                "created_at": "2026-01-07T00:00:00Z",
            },
            {
                "temperatura": 31.0,
                "umiditate": 63.0,
                "pm25": 65.0,
                "pm10": 105.0,
                "co2": 1380.0,
                "quality_label": "poor",
                "created_at": "2026-01-08T00:00:00Z",
            },
            {
                "temperatura": 26.0,
                "umiditate": 53.0,
                "pm25": 34.0,
                "pm10": 58.0,
                "co2": 930.0,
                "quality_label": "moderate",
                "created_at": "2026-01-09T00:00:00Z",
            },
            {
                "temperatura": 27.0,
                "umiditate": 55.0,
                "pm25": 38.0,
                "pm10": 62.0,
                "co2": 980.0,
                "quality_label": "moderate",
                "created_at": "2026-01-10T00:00:00Z",
            },
        ]
    )

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    try:
        train_model_module.load_training_data()
        assert False, "Expected RuntimeError for missing quality_label_source"
    except RuntimeError as exc:
        assert "quality_label_source" in str(exc)


def test_training_report_uses_database_labels_and_keeps_label_source(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_model.pkl"
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    _, report = train_model_module.train_and_save_random_forest(model_path=model_path, return_report=True)

    assert report["summary"]["label_source"] == "database_quality_label"


def test_training_report_uses_database_labels_when_available(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_model.pkl"

    rows = []
    for idx in range(36):
        rows.append(
            {
                "temperatura": 18.0 + (idx % 9),
                "umiditate": 35.0 + (idx % 15),
                "pm25": 8.0 + idx,
                "pm10": 18.0 + idx * 1.5,
                "co2": 500.0 + idx * 30,
                "quality_label": "good" if idx < 12 else "moderate" if idx < 24 else "poor",
                "quality_label_source": "manual",
                "created_at": f"2026-01-{(idx % 28) + 1:02d}T00:00:00Z",
            }
        )

    database_rows = pd.DataFrame(rows)
    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)

    _, report = train_model_module.train_and_save_random_forest(model_path=model_path, return_report=True)

    assert report["summary"]["label_source"] == "database_quality_label"
    assert report["evaluation"] is not None
    assert 0.0 <= report["evaluation"]["accuracy"] <= 1.0


def test_predict_air_quality_uses_latest_measurement(monkeypatch):
    database_rows = _sample_measurements_dataframe()

    class DummyModel:
        def predict(self, _input_df):
            return ["good"]

        def predict_proba(self, _input_df):
            return [[0.9, 0.1]]

    def fake_get_measurements(**kwargs):
        if kwargs.get("limit") == 1:
            return database_rows.iloc[[1]].reset_index(drop=True)
        return database_rows

    monkeypatch.setattr(predictor_module, "get_measurements", fake_get_measurements)
    monkeypatch.setattr(predictor_module, "load_model", lambda *args, **kwargs: DummyModel())

    prediction, confidence, feature_values, feature_assessment = predictor_module.predict_air_quality()

    assert prediction == "good"
    assert confidence == 0.9
    assert feature_values == {
        "temperature": 31.0,
        "humidity": 62.0,
        "pm25": 80.0,
        "pm10": 110.0,
        "co2": 1400.0,
    }
    assert set(feature_assessment.keys()) == {"temperature", "humidity", "pm25", "pm10", "co2"}
    assert feature_assessment["temperature"]["status"] == "poor"
    assert feature_assessment["humidity"]["status"] == "moderate"
    assert feature_assessment["pm25"]["status"] == "poor"
    assert feature_assessment["co2"]["condition"] == "poluat"


def test_build_forecast_returns_future_horizons(monkeypatch):
    database_rows = _sample_measurements_dataframe()

    class DummyModel:
        def predict(self, _input_df):
            return ["moderate"]

        def predict_proba(self, _input_df):
            return [[0.2, 0.8]]

    monkeypatch.setattr(predictor_module, "get_measurements", lambda **kwargs: database_rows)
    monkeypatch.setattr(predictor_module, "load_model", lambda *args, **kwargs: DummyModel())

    base_values = {
        "temperature": 25.0,
        "humidity": 50.0,
        "pm25": 20.0,
        "pm10": 35.0,
        "co2": 700.0,
    }

    forecast = predictor_module.build_forecast(
        base_feature_values=base_values,
        horizons_hours=[1, 3, 6],
        lookback_hours=168,
    )

    assert len(forecast) == 3
    assert [item["horizon_hours"] for item in forecast] == [1, 3, 6]
    assert all(item["prediction"] == "moderate" for item in forecast)
    assert all("input_values" in item for item in forecast)
    assert all("feature_assessment" in item for item in forecast)


def test_forecast_average_is_classified_from_future_values(monkeypatch):
    class DummyModel:
        def predict(self, input_df):
            assert input_df.iloc[0]["temperature"] == 25.0
            return ["moderate"]

        def predict_proba(self, _input_df):
            return [[0.3, 0.7]]

    monkeypatch.setattr(predictor_module, "load_model", lambda *args, **kwargs: DummyModel())
    average = predictor_module.summarize_forecast_average(
        [
            {
                "horizon_hours": 1,
                "input_values": {"temperature": 20.0, "humidity": 50.0, "pm25": 10.0, "pm10": 20.0, "co2": 600.0},
            },
            {
                "horizon_hours": 2,
                "input_values": {"temperature": 30.0, "humidity": 60.0, "pm25": 20.0, "pm10": 40.0, "co2": 800.0},
            },
        ]
    )

    assert average is not None
    assert average["prediction"] == "moderate"
    assert average["input_values"]["temperature"] == 25.0
    assert average["hours_averaged"] == [1, 2]


def test_august_temperature_forecast_limits_unrealistic_short_term_drop():
    forecast_timestamp = pd.Timestamp("2026-08-13T12:00:00Z")

    forecast_temperature = predictor_module._clamp_forecast_temperature(
        projected_value=9.0,
        current_value=30.0,
        forecast_timestamp=forecast_timestamp,
        horizon_hours=48,
    )

    assert forecast_temperature == 18.0


def test_temperature_hour_profile_cools_evening_forecast():
    measurements = pd.DataFrame(
        [
            {"created_at": "2026-08-01T09:00:00Z", "temperature": 30.0},
            {"created_at": "2026-08-01T18:00:00Z", "temperature": 20.0},
            {"created_at": "2026-08-02T09:00:00Z", "temperature": 30.0},
            {"created_at": "2026-08-02T18:00:00Z", "temperature": 20.0},
            {"created_at": "2026-08-03T09:00:00Z", "temperature": 30.0},
            {"created_at": "2026-08-03T18:00:00Z", "temperature": 20.0},
        ]
    )

    profile = predictor_module._compute_temperature_hour_profile(measurements)
    adjustment = predictor_module._temperature_calendar_adjustment(
        profile,
        pd.Timestamp("2026-08-04T09:00:00Z"),
        pd.Timestamp("2026-08-04T18:00:00Z"),
    )

    assert profile == {12: 30.0, 21: 20.0}
    assert adjustment == -10.0


def test_temperature_hour_profile_prioritizes_latest_two_days_same_hour_average():
    measurements = pd.DataFrame(
        [
            {"created_at": "2026-08-01T09:00:00Z", "temperature": 10.0},
            {"created_at": "2026-08-02T09:00:00Z", "temperature": 28.0},
            {"created_at": "2026-08-03T09:00:00Z", "temperature": 32.0},
        ]
    )

    profile = predictor_module._compute_temperature_hour_profile(measurements)

    assert profile == {12: 30.0}


def test_previous_hour_temperature_average_uses_only_the_preceding_full_hour():
    measurements = pd.DataFrame(
        [
            {"created_at": "2026-08-03T17:10:00Z", "temperature": 20.0},
            {"created_at": "2026-08-03T17:50:00Z", "temperature": 24.0},
            {"created_at": "2026-08-03T18:05:00Z", "temperature": 30.0},
        ]
    )

    previous_hour_average = predictor_module._previous_hour_temperature_average(
        measurements,
        pd.Timestamp("2026-08-03T18:05:00Z"),
    )

    assert previous_hour_average == 22.0


def test_temperature_diurnal_fallback_cools_evening_and_warms_afternoon():
    current_timestamp = pd.Timestamp("2026-08-04T17:00:00Z")  # 20:00 Romania

    evening_adjustment = predictor_module._temperature_calendar_adjustment(
        {},
        current_timestamp,
        pd.Timestamp("2026-08-04T19:00:00Z"),  # 22:00 Romania
    )
    afternoon_adjustment = predictor_module._temperature_calendar_adjustment(
        {},
        current_timestamp,
        pd.Timestamp("2026-08-05T13:00:00Z"),  # 16:00 Romania
    )

    assert evening_adjustment < 0
    assert afternoon_adjustment > 0


def test_temperature_forecast_uses_romanian_local_month():
    # 21:00 UTC on August 31 is September 1 in Romania (UTC+3).
    forecast_timestamp = pd.Timestamp("2026-08-31T21:00:00Z")

    forecast_temperature = predictor_module._clamp_forecast_temperature(
        projected_value=5.0,
        current_value=18.0,
        forecast_timestamp=forecast_timestamp,
        horizon_hours=48,
    )

    assert forecast_temperature == 6.0


def test_anomaly_detector_output_shape(tmp_path, monkeypatch):
    model_path = tmp_path / "air_quality_if.pkl"
    database_rows = _sample_measurements_dataframe()

    monkeypatch.setattr(train_model_module, "get_measurements", lambda **kwargs: database_rows)
    train_and_save_isolation_forest(model_path=model_path)

    from app.core import config

    original_if_model_path = config.IF_MODEL_PATH
    config.IF_MODEL_PATH = model_path
    monkeypatch.setattr(anomaly_detector_module, "get_measurements", lambda **kwargs: database_rows.iloc[[1]])

    try:
        result = detect_anomaly()
    finally:
        config.IF_MODEL_PATH = original_if_model_path

    assert "is_anomaly" in result
    assert "score" in result
    assert "input_values" in result
    assert "anomalous_features" in result
    assert "feature_analysis" in result
    assert result["label"] in {"anomaly", "normal"}
