#!/usr/bin/env python3
"""
Phase 2 data generator: gym workout log → NDJSON report.
Seeded RNG. 150 train + 15 valid + 30 test + 10 edge-case test.

DETERMINISTIC RULES (what the model must learn):
─────────────────────────────────────────────────
session_summary: always first line.
  session_rating 1-2: duration<20 OR "exhausted"/"terrible"/"rough" in notes.
  session_rating 4-5: any PR achieved OR "great"/"amazing" in notes.
  default rating: 3.

highlight records:
  PR_ACHIEVED      – any set has reps*weight > prev_pr[exercise].
  VOLUME_INCREASE  – curr_total_volume >= 1.05 * prev_total_volume.
  VOLUME_DECREASE  – curr_total_volume <= 0.95 * prev_total_volume.
  PROGRESS_NOTED   – weight increased for >=2 exercises vs prev, no big vol change.
  PLATEAU_DETECTED – "plateau" in notes OR same weight as prev + no rep gain.

recommendation records:
  REST_DAY_RECOMMENDED – duration<20 OR n_exercises<2 (checked first).
  NUTRITION_CHECK      – n_exercises>5 OR "energy low"/"hungry" in notes.
  MOBILITY_WORK        – "tight"/"sore"/"stiff" in notes.
  FORM_CHECK_NEEDED    – "form breaking"/"technique off" in notes.
  TECHNIQUE_REVIEW     – "unsure"/"new exercise" in notes.
  INCREASE_LOAD        – "felt easy"/"too easy"/"could do more" in notes.
  DECREASE_LOAD        – "too heavy"/"failed reps"/"missed" in notes.
  DELOAD_SUGGESTED     – rating<=2 AND duration<30.
  ADD_ACCESSORY_WORK   – all exercises are compound (no isolation) AND >=2 exercises.
  SUPERSET_SUGGESTED   – duration<45 AND n_exercises>=3 AND no other rec yet.
  MAINTAIN_LOAD        – no other recommendation was added.
"""
import json
import random
from pathlib import Path

SEED = 42
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS = ["Alex", "Jordan", "Sam", "Riley", "Morgan", "Casey"]
COMPOUND = ["Bench Press", "Squat", "Deadlift", "Overhead Press",
            "Barbell Row", "Pull-up", "Dip", "Romanian Deadlift"]
ISOLATION = ["Bicep Curl", "Tricep Pushdown", "Lateral Raise",
             "Face Pull", "Leg Curl", "Leg Extension", "Calf Raise"]
ALL_EX = COMPOUND + ISOLATION


def _make_session(rng: random.Random, edge: str | None = None) -> dict:
    user = rng.choice(USERS)
    year, month, day = 2024, rng.randint(1, 12), rng.randint(1, 28)
    date = f"{year}-{month:02d}-{day:02d}"

    if edge == "short":
        duration, n_ex = rng.randint(10, 18), 1
    elif edge == "long":
        duration, n_ex = rng.randint(95, 120), rng.randint(6, 8)
    else:
        duration, n_ex = rng.randint(30, 75), rng.randint(2, 5)

    ex_names = rng.sample(ALL_EX, min(n_ex, len(ALL_EX)))

    exercises = []
    prev_total = 0.0
    curr_total = 0.0
    has_isolation_ex = any(e in ISOLATION for e in ex_names)
    pr_achieved_any = False

    for name in ex_names:
        base_w = rng.choice([40, 50, 60, 70, 80, 90, 100, 110, 120])
        prev_w = base_w - rng.choice([0, 0, 2.5, 5])
        sets_count = rng.randint(2, 5)
        sets = []
        for _ in range(sets_count):
            reps = rng.randint(4, 12)
            w = base_w if rng.random() > 0.15 else base_w - 2.5
            sets.append({"reps": reps, "weight_kg": w})
            curr_total += reps * w

        prev_reps = rng.randint(4, 12)
        prev_total += prev_reps * prev_w * sets_count
        prev_pr = prev_reps * prev_w

        best_set = max(sets, key=lambda s: s["reps"] * s["weight_kg"])
        pr = best_set["reps"] * best_set["weight_kg"] > prev_pr * 1.02

        note = ""
        if edge == "missing":
            note = "" if rng.random() > 0.5 else rng.choice(["no notes", "n/a"])

        e = {
            "name": name,
            "sets": sets,
            "notes": note,
            "prev_weight_kg": prev_w,
            "weight_increased": base_w > prev_w,
            "pr_achieved": pr,
            "pr_weight": best_set["weight_kg"],
            "pr_reps": best_set["reps"],
        }
        if pr:
            pr_achieved_any = True
        exercises.append(e)

    session_notes = ""
    note_pool = [
        "", "", "",
        "felt easy on everything", "too easy, need more weight", "could do more",
        "felt heavy today", "missed last rep on deadlift", "too heavy",
        "form breaking on squat", "technique off on bench",
        "hips tight", "lower back sore", "shoulders stiff",
        "exhausted", "terrible energy", "rough day",
        "great session", "amazing energy",
        "plateau on squat again", "same weights as last time, no progress",
        "plateau", "new exercise, unsure about form",
        "energy low", "hungry during session",
        "good workout", "solid session",
    ]
    if edge == "missing":
        session_notes = ""
    else:
        session_notes = rng.choice(note_pool)

    return {
        "user": user,
        "date": date,
        "duration_min": duration,
        "exercises": exercises,
        "session_notes": session_notes,
        "prev_total_volume": prev_total,
        "curr_total_volume": curr_total,
        "pr_achieved_any": pr_achieved_any,
        "has_isolation": has_isolation_ex,
        "plateau_exercise": exercises[0]["name"] if exercises else "overall",
    }


def _apply_rules(s: dict) -> list[dict]:
    notes = " ".join([e["notes"] for e in s["exercises"]] + [s["session_notes"]]).lower()
    dur = s["duration_min"]
    n_ex = len(s["exercises"])
    cv, pv = s["curr_total_volume"], s["prev_total_volume"]
    vol_delta = (cv - pv) / max(pv, 1.0)

    rating = 3
    if dur < 20 or any(kw in notes for kw in ["exhausted", "terrible", "rough"]):
        rating = 2
    elif s["pr_achieved_any"] or any(kw in notes for kw in ["great", "amazing"]):
        rating = 5
    elif any(kw in notes for kw in ["good", "solid"]):
        rating = 4
    elif any(kw in notes for kw in ["tired", "heavy"]):
        rating = 2

    records: list[dict] = [{
        "record_type": "session_summary",
        "date": s["date"], "user": s["user"],
        "total_volume_kg": round(cv, 1),
        "exercise_count": n_ex, "session_rating": rating,
    }]

    for e in s["exercises"]:
        if e["pr_achieved"]:
            records.append({"record_type": "highlight", "action_id": "PR_ACHIEVED",
                "exercise": e["name"],
                "detail": f"New PR: {e['pr_weight']}kg x {e['pr_reps']} reps"})

    if vol_delta >= 0.05:
        records.append({"record_type": "highlight", "action_id": "VOLUME_INCREASE",
            "exercise": "overall",
            "detail": f"Total volume up {vol_delta*100:.0f}% vs last session"})
    elif vol_delta <= -0.05:
        records.append({"record_type": "highlight", "action_id": "VOLUME_DECREASE",
            "exercise": "overall",
            "detail": f"Total volume down {abs(vol_delta)*100:.0f}% vs last session"})

    w_increases = sum(1 for e in s["exercises"] if e["weight_increased"])
    if w_increases >= 2 and abs(vol_delta) < 0.05:
        records.append({"record_type": "highlight", "action_id": "PROGRESS_NOTED",
            "exercise": "multiple",
            "detail": f"Weight increased on {w_increases} exercises"})

    if "plateau" in notes:
        records.append({"record_type": "highlight", "action_id": "PLATEAU_DETECTED",
            "exercise": s["plateau_exercise"],
            "detail": "Same weight as previous session, no rep progression"})

    added: set[str] = set()
    def rec(action_id, exercise, reason, priority="medium"):
        if action_id not in added:
            added.add(action_id)
            records.append({"record_type": "recommendation", "action_id": action_id,
                "exercise": exercise, "reason": reason, "priority": priority})

    if dur < 20 or n_ex < 2:
        rec("REST_DAY_RECOMMENDED", "overall", "Session too short (<20 min or <2 exercises)", "high")
    if n_ex > 5 or any(kw in notes for kw in ["energy low", "hungry"]):
        rec("NUTRITION_CHECK", "overall", "Long session or low energy detected", "medium")
    if any(kw in notes for kw in ["tight", "sore", "stiff"]):
        rec("MOBILITY_WORK", "overall", "Tightness or soreness noted", "medium")
    if any(kw in notes for kw in ["form breaking", "technique off"]):
        rec("FORM_CHECK_NEEDED", "overall", "Form breakdown noted", "high")
    if any(kw in notes for kw in ["unsure", "new exercise"]):
        rec("TECHNIQUE_REVIEW", "overall", "Unfamiliar movement or form uncertainty", "medium")
    if any(kw in notes for kw in ["felt easy", "too easy", "could do more"]):
        rec("INCREASE_LOAD", "overall", "User reported session felt easy", "medium")
    if any(kw in notes for kw in ["too heavy", "failed", "missed"]):
        rec("DECREASE_LOAD", "overall", "Failed reps or load too heavy", "high")
    if rating <= 2 and dur < 30:
        rec("DELOAD_SUGGESTED", "overall", "Low rating and short session indicate fatigue", "high")
    if not s["has_isolation"] and n_ex >= 2:
        rec("ADD_ACCESSORY_WORK", "overall", "Session contains only compound movements", "low")
    if dur < 45 and n_ex >= 3 and not added:
        rec("SUPERSET_SUGGESTED", "overall",
            "Short session with multiple exercises; supersets could increase density", "low")
    if not added:
        rec("MAINTAIN_LOAD", "overall", "No clear signals for load adjustment", "low")

    return records


def _format_session(s: dict, style: str, rng: random.Random) -> str:
    exs = s["exercises"]
    def sets_str(e):
        return " | ".join(f"{st['reps']}x{st['weight_kg']}kg" for st in e["sets"])

    if style == "bullet":
        lines = [f"Workout Log — {s['user']} — {s['date']}",
                 f"Duration: {s['duration_min']} min", ""]
        for e in exs:
            line = f"• {e['name']}: {sets_str(e)}"
            if e["notes"]:
                line += f"  ({e['notes']})"
            lines.append(line)
        if s["session_notes"]:
            lines += ["", f"Notes: {s['session_notes']}"]
        return "\n".join(lines)

    if style == "prose":
        ex_parts = []
        for e in exs:
            sp = " then ".join(
                f"{st['reps']} reps at {st['weight_kg']}kg" for st in e["sets"])
            part = f"{e['name']} for {sp}"
            if e["notes"]:
                part += f", {e['notes']}"
            ex_parts.append(part)
        body = f"{s['user']} trained on {s['date']} for {s['duration_min']} minutes. "
        body += "Did " + "; ".join(ex_parts) + "."
        if s["session_notes"]:
            body += f" {s['session_notes'].capitalize()}."
        return body

    # terse
    lines = [f"{s['user']} {s['date']} {s['duration_min']}m"]
    abbrev = {"Bench Press": "BP", "Squat": "SQ", "Deadlift": "DL",
              "Overhead Press": "OHP", "Barbell Row": "BR", "Pull-up": "PU",
              "Dip": "DIP", "Romanian Deadlift": "RDL"}
    for e in exs:
        label = abbrev.get(e["name"], e["name"][:3].upper())
        s_str = "+".join(f"{st['reps']}@{int(st['weight_kg'])}" for st in e["sets"])
        note_part = f" ({e['notes']})" if e["notes"] else ""
        lines.append(f"{label}: {s_str}{note_part}")
    if s["session_notes"]:
        lines.append(f"Notes: {s['session_notes']}")
    return "\n".join(lines)


def _make_example(session: dict, rng: random.Random) -> dict:
    style = rng.choice(["bullet", "prose", "terse"])
    inp = _format_session(session, style, rng)
    records = _apply_rules(session)
    output_ndjson = "\n".join(json.dumps(r) for r in records)
    text = (f"<start_of_turn>user\n{inp}<end_of_turn>\n"
            f"<start_of_turn>model\n{output_ndjson}<end_of_turn>")
    return {"text": text, "input": inp, "expected_records": records}


def main():
    rng = random.Random(SEED)

    pool = [_make_example(_make_session(rng), rng) for _ in range(165)]
    rng.shuffle(pool)
    train_data, valid_data = pool[:150], pool[150:]

    test_normal = [_make_example(_make_session(rng), rng) for _ in range(30)]
    edges = []
    for etype in ["short"] * 4 + ["long"] * 3 + ["missing"] * 3:
        edges.append(_make_example(_make_session(rng, edge=etype), rng))
    test_data = test_normal + edges

    def write(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"]}) + "\n")

    def write_test(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write(DATA_DIR / "train.jsonl", train_data)
    write(DATA_DIR / "valid.jsonl", valid_data)
    write_test(DATA_DIR / "test.jsonl", test_data)

    print(f"train.jsonl : {len(train_data)}")
    print(f"valid.jsonl : {len(valid_data)}")
    print(f"test.jsonl  : {len(test_data)} ({len(test_normal)} normal + {len(edges)} edge)")
    print("\n--- First 3 train examples (input only) ---")
    for ex in train_data[:3]:
        print("INPUT:", ex["input"][:120], "...")
        print("NDJSON lines:", len(ex["expected_records"]))
        print()


if __name__ == "__main__":
    main()
