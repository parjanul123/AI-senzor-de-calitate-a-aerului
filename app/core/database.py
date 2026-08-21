import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import Client, create_client


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
_MISSING_CREDENTIALS_REPORTED = False
_CLIENT_CREATION_ERROR_REPORTED = False

DEVICE_ID_CANDIDATES = [
    "device_id",
    "deviceId",
    "sensor_id",
    "sensorId",
    "device_name",
    "name",
]
TIMESTAMP_CANDIDATES = ["created_at", "timestamp", "time", "recorded_at"]
LATITUDE_CANDIDATES = ["latitude", "lat", "gps_lat", "device_latitude"]
LONGITUDE_CANDIDATES = ["longitude", "lon", "lng", "gps_lng", "device_longitude"]
LOCATION_TEXT_CANDIDATES = [
    "location",
    "location_name",
    "locatie",
    "city",
    "address",
    "device_location",
]
LOCATION_TABLE_CANDIDATES = [
    "location",
    "locations",
    "device_locations",
    "device_location",
    "devices",
    "device",
]
LOCATION_DEVICE_ID_CANDIDATES = [
    "device_identifier",
    "device_id",
    "deviceId",
    "sensor_id",
    "sensorId",
    "name",
    "device_name",
    "id",
]
MEASUREMENT_DEVICE_REF_CANDIDATES = [
    "device_id",
    "deviceId",
    "sensor_id",
    "sensorId",
    "device_ref_id",
]


def _create_supabase_client() -> Optional[Client]:
    """Create a Supabase client using environment variables from .env."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_service_role_key:
        global _MISSING_CREDENTIALS_REPORTED
        missing_vars = []
        if not supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not supabase_service_role_key:
            missing_vars.append("SUPABASE_SERVICE_ROLE_KEY")
        message = (
            f"Missing Supabase credentials: {', '.join(missing_vars)}. "
            f"Expected values in {ENV_PATH}."
        )
        if not _MISSING_CREDENTIALS_REPORTED:
            logger.warning(message)
            _MISSING_CREDENTIALS_REPORTED = True
        else:
            logger.debug(message)
        return None

    try:
        return create_client(supabase_url, supabase_service_role_key)
    except Exception as exc:
        global _CLIENT_CREATION_ERROR_REPORTED
        message = f"Failed to create Supabase client: {exc}"
        # Keep this as debug-only to avoid noisy stderr output in local runs.
        # The app already exposes Supabase config diagnostics in the UI.
        if not _CLIENT_CREATION_ERROR_REPORTED:
            logger.debug(message)
            _CLIENT_CREATION_ERROR_REPORTED = True
        else:
            logger.debug(message)
        return None


def _detect_existing_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _normalize_measurements_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    normalized = dataframe.copy()

    if "device" in normalized.columns:
        normalized["device_name"] = normalized["device"].apply(
            lambda value: value.get("name") if isinstance(value, dict) else None
        )
        normalized["device_ref_id"] = normalized["device"].apply(
            lambda value: value.get("id") if isinstance(value, dict) else None
        )
        normalized["device_location"] = normalized["device"].apply(
            lambda value: value.get("location") if isinstance(value, dict) else None
        )
        normalized["device_latitude"] = normalized["device"].apply(
            lambda value: value.get("latitude") if isinstance(value, dict) else None
        )
        normalized["device_longitude"] = normalized["device"].apply(
            lambda value: value.get("longitude") if isinstance(value, dict) else None
        )

    existing_columns = list(normalized.columns)

    device_column = _detect_existing_column(existing_columns, DEVICE_ID_CANDIDATES)
    if device_column is not None:
        normalized["device_identifier"] = normalized[device_column].astype(str)
    elif "device_name" in normalized.columns:
        normalized["device_identifier"] = normalized["device_name"].astype(str)
    elif "device_ref_id" in normalized.columns:
        normalized["device_identifier"] = normalized["device_ref_id"].astype(str)

    timestamp_column = _detect_existing_column(existing_columns, TIMESTAMP_CANDIDATES)
    if timestamp_column is not None:
        normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], errors="coerce")

    return normalized


def _sort_measurements(dataframe: pd.DataFrame, descending: bool = True) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    timestamp_column = _detect_existing_column(list(dataframe.columns), TIMESTAMP_CANDIDATES)
    if timestamp_column is None:
        return dataframe

    return dataframe.sort_values(timestamp_column, ascending=not descending)


def _fetch_measurements_via_rest(limit: Optional[int] = None, descending: bool = True) -> list[dict[str, Any]]:
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_service_role_key:
        return []

    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/measurements"
    headers = {
        "apikey": supabase_service_role_key,
        "Authorization": f"Bearer {supabase_service_role_key}",
        "Content-Type": "application/json",
    }

    select_candidates = ["*,device(*)", "*"]

    if limit is None:
        for select_value in select_candidates:
            try:
                response = requests.get(
                    endpoint,
                    params={"select": select_value},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                continue
        return []

    for select_value in select_candidates:
        for order_column in ["created_at", "timestamp", "time", "recorded_at"]:
            params = {
                "select": select_value,
                "limit": limit,
                "order": f"{order_column}.{'desc' if descending else 'asc'}",
            }
            try:
                response = requests.get(endpoint, params=params, headers=headers, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                continue

    for select_value in select_candidates:
        try:
            response = requests.get(
                endpoint,
                params={"select": select_value, "limit": limit},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            continue

    return []


def _fetch_measurements_records(client: Client, limit: Optional[int], descending: bool) -> list[dict[str, Any]]:
    select_candidates = ["*,device(*)", "*"]

    if limit is None:
        for select_value in select_candidates:
            try:
                response = client.table("measurements").select(select_value).execute()
                return response.data or []
            except Exception:
                continue
        return []

    for select_value in select_candidates:
        for order_column in ["created_at", "timestamp", "time", "recorded_at"]:
            try:
                response = (
                    client.table("measurements")
                    .select(select_value)
                    .order(order_column, desc=descending)
                    .limit(limit)
                    .execute()
                )
                return response.data or []
            except Exception:
                continue

    for select_value in select_candidates:
        try:
            response = client.table("measurements").select(select_value).limit(limit).execute()
            return response.data or []
        except Exception:
            continue

    return []


def get_measurements(
    device_identifier: Optional[str] = None,
    limit: Optional[int] = None,
    descending: bool = True,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    # When filtering by device the limit must be applied after filtering, otherwise the
    # newest N global rows may contain no row for the requested device.
    fetch_limit = None if device_identifier else limit

    client = _create_supabase_client()
    if client is None:
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            if raise_on_error:
                raise RuntimeError(
                    "Lipsesc variabilele SUPABASE_URL sau SUPABASE_SERVICE_ROLE_KEY. "
                    "Configurează-le în serviciul Railway."
                )
            return pd.DataFrame()
        try:
            records = _fetch_measurements_via_rest(limit=fetch_limit, descending=descending)
        except Exception as exc:
            logger.exception("Failed to fetch data from Supabase table 'measurements' via REST fallback: %s", exc)
            if raise_on_error:
                raise RuntimeError(
                    "Conexiunea Supabase nu a putut fi creată. Verifică SUPABASE_URL și SUPABASE_SERVICE_ROLE_KEY în .env."
                ) from exc
            return pd.DataFrame()
    else:
        try:
            records = _fetch_measurements_records(client, limit=fetch_limit, descending=descending)
        except Exception as exc:
            logger.exception("Failed to fetch data from Supabase table 'measurements': %s", exc)
            if raise_on_error:
                raise RuntimeError(f"Interogarea în tabela 'measurements' a eșuat: {exc}") from exc
            return pd.DataFrame()

    if not records:
        logger.warning("No rows found in Supabase table 'measurements'.")
        return pd.DataFrame()

    dataframe = pd.DataFrame.from_records(records)
    dataframe = _normalize_measurements_dataframe(dataframe)
    dataframe = _sort_measurements(dataframe, descending=descending)

    if device_identifier:
        if "device_identifier" not in dataframe.columns:
            logger.warning(
                "Requested device filter '%s', but no device identifier column exists.",
                device_identifier,
            )
            return pd.DataFrame.from_records([])

        requested = str(device_identifier).strip().lower()
        dataframe = dataframe[
            dataframe["device_identifier"].astype(str).str.strip().str.lower() == requested
        ]
        if limit is not None:
            dataframe = dataframe.head(int(limit))

    return dataframe.reset_index(drop=True)


def get_device_identifiers() -> list[str]:
    dataframe = get_measurements()
    if dataframe.empty or "device_identifier" not in dataframe.columns:
        return []

    devices = dataframe["device_identifier"].dropna().astype(str).str.strip().replace("", pd.NA).dropna()
    return sorted(devices.unique().tolist())


def _pick_first_non_empty(row: pd.Series, candidates: list[str]) -> Any:
    for column in candidates:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_location_payload(record: dict[str, Any], source_table: str, source_id_column: str) -> Optional[dict[str, Any]]:
    if not record:
        return None

    location_text = None
    for key in LOCATION_TEXT_CANDIDATES:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            location_text = str(value).strip()
            break

    latitude = None
    for key in LATITUDE_CANDIDATES:
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            latitude = float(value)
            break
        except (TypeError, ValueError):
            continue

    longitude = None
    for key in LONGITUDE_CANDIDATES:
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            longitude = float(value)
            break
        except (TypeError, ValueError):
            continue

    if location_text is None and latitude is None and longitude is None:
        return None

    return {
        "location": location_text,
        "latitude": latitude,
        "longitude": longitude,
        "source_table": source_table,
        "source_id_column": source_id_column,
    }


def _lookup_location_via_client(client: Client, identifiers: list[str]) -> Optional[dict[str, Any]]:
    for table_name in LOCATION_TABLE_CANDIDATES:
        for id_column in LOCATION_DEVICE_ID_CANDIDATES:
            for identifier_value in identifiers:
                try:
                    response = (
                        client.table(table_name)
                        .select("*")
                        .eq(id_column, identifier_value)
                        .limit(1)
                        .execute()
                    )
                except Exception:
                    continue

                rows = response.data or []
                if not rows:
                    continue

                payload = _extract_location_payload(rows[0], source_table=table_name, source_id_column=id_column)
                if payload:
                    return payload

    return None


def _lookup_location_via_rest(identifiers: list[str]) -> Optional[dict[str, Any]]:
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_service_role_key:
        return None

    headers = {
        "apikey": supabase_service_role_key,
        "Authorization": f"Bearer {supabase_service_role_key}",
        "Content-Type": "application/json",
    }

    for table_name in LOCATION_TABLE_CANDIDATES:
        endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}"
        for id_column in LOCATION_DEVICE_ID_CANDIDATES:
            for identifier_value in identifiers:
                try:
                    response = requests.get(
                        endpoint,
                        params={"select": "*", id_column: f"eq.{identifier_value}", "limit": 1},
                        headers=headers,
                        timeout=20,
                    )
                    response.raise_for_status()
                    rows = response.json() or []
                except (requests.RequestException, ValueError):
                    continue

                if not rows:
                    continue

                payload = _extract_location_payload(rows[0], source_table=table_name, source_id_column=id_column)
                if payload:
                    return payload

    return None


def get_device_location_details(
    device_identifier: str | None,
    aliases: list[str] | None = None,
) -> Optional[dict[str, Any]]:
    requested_identifier = (device_identifier or "").strip()
    if not requested_identifier and not aliases:
        return None

    identifier_candidates: list[str] = []
    if requested_identifier:
        identifier_candidates.append(requested_identifier)
    for alias in aliases or []:
        alias_value = str(alias).strip()
        if alias_value and alias_value not in identifier_candidates:
            identifier_candidates.append(alias_value)

    if not identifier_candidates:
        return None

    # Fast path: if measurements already include location, use the newest row.
    dataframe = get_measurements(device_identifier=requested_identifier or identifier_candidates[0], limit=1, descending=True)
    if not dataframe.empty:
        row = dataframe.iloc[0].to_dict()

        for ref_column in MEASUREMENT_DEVICE_REF_CANDIDATES:
            ref_value = row.get(ref_column)
            if ref_value is None:
                continue
            ref_text = str(ref_value).strip()
            if ref_text and ref_text not in identifier_candidates:
                identifier_candidates.append(ref_text)

        payload = _extract_location_payload(
            row,
            source_table="measurements",
            source_id_column="device_identifier",
        )
        if payload:
            return payload

    client = _create_supabase_client()
    if client is not None:
        payload = _lookup_location_via_client(client, identifier_candidates)
        if payload:
            return payload

    return _lookup_location_via_rest(identifier_candidates)


def get_devices_with_location() -> list[dict[str, Any]]:
    dataframe = get_measurements(descending=True)
    if dataframe.empty or "device_identifier" not in dataframe.columns:
        return []

    timestamp_column = _detect_existing_column(list(dataframe.columns), TIMESTAMP_CANDIDATES)
    if timestamp_column:
        dataframe = dataframe.sort_values(timestamp_column, ascending=False)

    latest_per_device = dataframe.dropna(subset=["device_identifier"]).copy()
    latest_per_device["device_identifier"] = latest_per_device["device_identifier"].astype(str).str.strip()
    latest_per_device = latest_per_device[latest_per_device["device_identifier"] != ""]
    latest_per_device = latest_per_device.drop_duplicates(subset=["device_identifier"], keep="first")

    devices: list[dict[str, Any]] = []
    for _, row in latest_per_device.iterrows():
        device_identifier = str(row.get("device_identifier", "")).strip()
        if not device_identifier:
            continue

        latitude = _to_float_or_none(_pick_first_non_empty(row, LATITUDE_CANDIDATES))
        longitude = _to_float_or_none(_pick_first_non_empty(row, LONGITUDE_CANDIDATES))
        location_text_raw = _pick_first_non_empty(row, LOCATION_TEXT_CANDIDATES)
        location_text = str(location_text_raw).strip() if location_text_raw is not None else None
        if location_text == "":
            location_text = None

        if location_text and latitude is not None and longitude is not None:
            location_label = f"{location_text} ({latitude:.5f}, {longitude:.5f})"
        elif location_text:
            location_label = location_text
        elif latitude is not None and longitude is not None:
            location_label = f"{latitude:.5f}, {longitude:.5f}"
        else:
            location_label = "Locație necunoscută"

        last_seen = row.get(timestamp_column) if timestamp_column else None
        last_seen_iso = None
        if timestamp_column and pd.notna(last_seen):
            try:
                last_seen_iso = pd.to_datetime(last_seen).isoformat()
            except (TypeError, ValueError):
                last_seen_iso = str(last_seen)

        devices.append(
            {
                "device_identifier": device_identifier,
                "location": location_text,
                "latitude": latitude,
                "longitude": longitude,
                "location_label": location_label,
                "last_seen": last_seen_iso,
            }
        )

    return sorted(devices, key=lambda item: item["device_identifier"].lower())


def summarize_measurements(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "preview": dataframe.head(5),
    }


def get_supabase_config_status() -> dict[str, Any]:
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    return {
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
        "supabase_url_set": bool(url),
        "service_role_key_set": bool(key),
    }
