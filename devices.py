"""
Вибір QPU для платних запусків на Amazon Braket і журнал запусків.

Пристрій обирається змінною середовища BRAKET_DEVICE (за замовчуванням garnet):
  garnet  — IQM Garnet, 20 кубітів, квадратна решітка, eu-north-1, $0.30 + $0.00145/shot (наш базовий)
  emerald — IQM Emerald, 54 кубіти, квадратна решітка, eu-north-1, $0.30 + $0.0016/shot (точність краща за Garnet)
  forte   — IonQ Forte-1, 36 кубітів, іонна пастка з ПОВНОЮ зв'язністю, us-east-1, $0.30 + $0.08/shot
            (1000 shots ≈ $80 — перед submit запитується підтвердження; з error mitigation IonQ вимагає ≥ 2500 shots)
Ціни — з https://aws.amazon.com/braket/pricing/ (перевірено 2026-11-07); ARN — з консолі Braket, якщо
пристрій не знайдено, скрипт друкує список доступних у регіоні.

Усі платні submit і fetch дописуються в qpu_runs.json (пристрій, Task ID, скрипт, інстанс, shots, оцінка
вартості, параметри, а після fetch — результат), щоб журнал витрат і результатів був в одному місці.
"""
import json, os, sys, datetime

DEVICES = {
    "garnet": dict(name="IQM Garnet", arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet", region="eu-north-1",
                   qubits=20, connectivity="lattice", per_task=0.30, per_shot=0.00145, confirm_above_usd=10),
    "emerald": dict(name="IQM Emerald", arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald", region="eu-north-1",
                    qubits=54, connectivity="lattice", per_task=0.30, per_shot=0.0016, confirm_above_usd=10),
    "forte": dict(name="IonQ Forte-1", arn="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1", region="us-east-1",
                  qubits=36, connectivity="all-to-all", per_task=0.30, per_shot=0.08, confirm_above_usd=0),
}
LOG = "qpu_runs.json"


def selected():
    key = os.environ.get("BRAKET_DEVICE", "garnet").lower()
    if key not in DEVICES:
        sys.exit(f"BRAKET_DEVICE={key!r} невідомий; доступні: {', '.join(DEVICES)}")
    return key, DEVICES[key]


def cost(dev, shots):
    return dev["per_task"] + shots * dev["per_shot"]


def session(dev):
    import boto3
    from braket.aws import AwsSession
    return AwsSession(boto3.Session(region_name=dev["region"]))


def open_device(dev, sess):
    """AwsDevice або зрозуміла помилка зі списком пристроїв регіону."""
    from braket.aws import AwsDevice
    try:
        return AwsDevice(dev["arn"], aws_session=sess)
    except Exception as e:
        print(f"Не вдалося відкрити {dev['name']} ({dev['arn']}): {e}")
        try:
            print("Пристрої в регіоні", dev["region"] + ":")
            for d in AwsDevice.get_devices(aws_session=sess, types=["QPU"]):
                print("  ", d.arn, d.status)
        except Exception as e2:
            print("  список недоступний:", e2)
        sys.exit(1)


def check_ready(device, n_qubits_needed):
    """Статус, кількість кубітів і вікна доступності. ONLINE ≠ виконує зараз (урок 2026-10-04)."""
    print(f"Пристрій: {device.name}, статус {device.status}")
    try:
        props = device.properties
        qc = getattr(getattr(props, "paradigm", None), "qubitCount", None)
        if qc is not None:
            print(f"  кубітів: {qc} (потрібно {n_qubits_needed})")
            if n_qubits_needed > qc:
                sys.exit("  задача не влазить у пристрій")
        windows = props.service.executionWindows
        now = datetime.datetime.now(datetime.timezone.utc)
        print(f"  зараз {now:%a %H:%M} UTC; вікна доступності:")
        for w in windows:
            print(f"    {w.executionDay}: {w.windowStartHour}–{w.windowEndHour} UTC")
    except Exception as e:
        print("  властивості недоступні:", e)
    if device.status != "ONLINE":
        print("  УВАГА: пристрій не ONLINE — задача стоятиме в черзі")


def confirm(dev, shots):
    c = cost(dev, shots)
    print(f"Відправка на {dev['name']}: {shots} shots, орієнтовна вартість ≈ ${c:.2f}")
    if c > dev["confirm_above_usd"] or dev["per_shot"] >= 0.01:
        if os.environ.get("BRAKET_YES") != "1":
            ans = input(f"  Підтвердити платний запуск за ≈${c:.2f}? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit("  скасовано")
    return c


def log_submit(task_id, dev_key, script, shots, est_cost, **params):
    runs = json.load(open(LOG, encoding="utf-8")) if os.path.exists(LOG) else []
    runs.append(dict(task_id=task_id, device=dev_key, script=script, shots=shots, est_cost_usd=round(est_cost, 3),
                     submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     params=params, result=None))
    json.dump(runs, open(LOG, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def log_result(task_id, result):
    if not os.path.exists(LOG):
        return
    runs = json.load(open(LOG, encoding="utf-8"))
    for r in runs:
        if r["task_id"] == task_id or task_id.endswith(r["task_id"]) or r["task_id"].endswith(task_id):
            r["result"] = result
            r["fetched_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    json.dump(runs, open(LOG, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def kind_of(task_id):
    """'control', якщо Task ID записано як контроль у журналі, інакше 'main'."""
    if os.path.exists(LOG):
        for r in json.load(open(LOG, encoding="utf-8")):
            if r["task_id"] == task_id and r["params"].get("control"):
                return "control"
    return "main"


if __name__ == "__main__":
    key, dev = selected()
    print(key, json.dumps(dev, indent=2, ensure_ascii=False))
    for s in (700, 1000, 2500):
        print(f"  {s} shots ≈ ${cost(dev, s):.2f}")
