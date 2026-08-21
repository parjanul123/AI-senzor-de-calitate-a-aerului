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

    if limit is None:
        response = requests.get(endpoint, params={"select": "*"}, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()

    for order_column in ["created_at", "timestamp", "time", "recorded_at"]:
        params = {
            "select": "*",
            "limit": limit,
            "order": f"{order_column}.{'desc' if descending else 'asc'}",
        }
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            continue

    response = requests.get(endpoint, params={"select": "*", "limit": limit}, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _fetch_measurements_records(client: Client, limit: Optional[int], descending: bool) -> list[dict[str, Any]]:
    if limit is None:
        response = client.table("measurements").select("*").execute()
        return response.data or []

    for order_column in ["created_at", "timestamp", "time", "recorded_at"]:
        try:
            response = (
                client.table("measurements")
                .select("*")
                .order(order_column, desc=descending)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception:
            continue

    response = client.table("measurements").select("*").limit(limit).execute()
    return response.data or []


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
