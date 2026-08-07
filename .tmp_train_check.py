import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

repo = r"D:\AI senzor de temperatura"
py = os.path.join(repo, ".venv", "Scripts", "python.exe")
proc = subprocess.Popen([py, "-m", "uvicorn", "app.api.main:app", "--port", "8017"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for_server(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8017/openapi.json", timeout=5) as r:
                return r.read().decode("utf-8")
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def find_dataset_info(obj):
    if isinstance(obj, dict):
        if "dataset_info" in obj and isinstance(obj["dataset_info"], dict):
            return obj["dataset_info"]
        for value in obj.values():
            found = find_dataset_info(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_dataset_info(item)
            if found is not None:
                return found
    return None


def try_request(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8017" + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"HTTP_ERROR {e.code}: {body}")
        return None
    except Exception as e:
        print(f"REQUEST_ERROR: {e}")
        return None


try:
    openapi_text = wait_for_server()
    openapi = json.loads(openapi_text)
    candidate_paths = []
    for path, methods in openapi.get("paths", {}).items():
        if isinstance(methods, dict) and "post" in methods and "train" in path.lower():
            candidate_paths.append(path)
    if not candidate_paths:
        raise RuntimeError(f"no training endpoint paths in openapi: {list(openapi.get('paths', {}).keys())}")
    print("CANDIDATE_PATHS:", candidate_paths)
    payloads = [
        {"model_name": "random_forest", "hourly_aggregation": False},
        {"hourly_aggregation": False},
    ]
    response = None
    for path in candidate_paths:
        for payload in payloads:
            response = try_request(path, payload)
            if response is not None:
                break
        if response is not None:
            break
    if response is None:
        raise RuntimeError("no successful response")
    dataset_info = find_dataset_info(response)
    if dataset_info is None:
        raise RuntimeError(f"dataset_info not found in response: {json.dumps(response)[:2000]}")
    print("PATH:", path)
    print("DATASET_INFO:")
    print(json.dumps({k: dataset_info.get(k) for k in ["total_measurements", "device_count", "time_range"]}, indent=2))
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)
    elif proc.stdout is not None:
        output = proc.stdout.read()
        if output:
            print(output)
