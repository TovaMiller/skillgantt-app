
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, date
import json, re, copy
from collections import defaultdict, deque

st.set_page_config(page_title="SkillGantt v6.1", layout="wide")

# ---- (reuse full v6 code but with updated home menu) ----
# For brevity, we import the v6 code by embedding it here with tweaks.

SKILL_FACTORS = {5:1.0, 4:1.2, 3:1.5, 2:2.5}

COUNTRY_HOLIDAYS = {
    "Italy-2025": ["2025-01-01","2025-01-06","2025-04-20","2025-04-21","2025-04-25","2025-05-01","2025-06-02","2025-08-15","2025-11-01","2025-12-08","2025-12-25","2025-12-26"],
    "United Kingdom-2025": ["2025-01-01","2025-04-18","2025-04-21","2025-05-05","2025-05-26","2025-08-25","2025-12-25","2025-12-26"],
    "United States-2025": ["2025-01-01","2025-01-20","2025-02-17","2025-05-26","2025-06-19","2025-07-04","2025-09-01","2025-10-13","2025-11-11","2025-11-27","2025-12-25"],
    "Germany-2025": ["2025-01-01","2025-04-18","2025-04-21","2025-05-01","2025-05-29","2025-06-09","2025-10-03","2025-12-25","2025-12-26"],
}

def parse_ics_dates(text: str):
    dates = set()
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"DTSTART(?:;VALUE=DATE)?:(\d{8})", line)
        if m:
            ymd = m.group(1)
            dates.add(f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}")
    return dates

def to_date(x):
    if isinstance(x, str):
        return datetime.fromisoformat(x).date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    return x

def load_state():
    ss = st.session_state
    ss.setdefault("engineers", [])
    ss.setdefault("tasks", [])
    ss.setdefault("allocations", {})
    ss.setdefault("baseline_allocations", {})
    ss.setdefault("workweek", {0:True,1:True,2:True,3:True,4:True,5:False,6:False})
    ss.setdefault("holidays", set())
    ss.setdefault("dirty", False)
    ss.setdefault("scenarios", [])
    ss.setdefault("whatif_last", {})

def reset_all():
    st.session_state.engineers = []
    st.session_state.tasks = []
    st.session_state.allocations = {}
    st.session_state.baseline_allocations = {}
    st.session_state.scenarios = []
    st.session_state.whatif_last = {}
    st.session_state.dirty = True

def serialize_alloc(obj):
    return {str(k): {
        "engineer_id": v["engineer_id"],
        "start": v["start"].isoformat(),
        "end": v["end"].isoformat(),
        "duration_days": v["duration_days"],
        "factor": v.get("factor",1.0),
        "capacity_pct": v.get("capacity_pct",100),
        "eff_duration_days": v.get("eff_duration_days", v["duration_days"]),
    } for k, v in obj.items()}

def parse_alloc(obj):
    parsed = {}
    for k, v in obj.items():
        parsed[int(k)] = {
            "engineer_id": v["engineer_id"],
            "start": datetime.fromisoformat(v["start"]),
            "end": datetime.fromisoformat(v["end"]),
            "duration_days": v["duration_days"],
            "factor": v.get("factor",1.0),
            "capacity_pct": v.get("capacity_pct", 100),
            "eff_duration_days": v.get("eff_duration_days", v["duration_days"]),
        }
    return parsed

def export_json() -> str:
    data = {
        "engineers": [{**e, "time_off": sorted(list(e.get("time_off", set())))} for e in st.session_state.engineers],
        "tasks": st.session_state.tasks,
        "allocations": serialize_alloc(st.session_state.allocations),
        "baseline_allocations": serialize_alloc(st.session_state.baseline_allocations),
        "settings": {"workweek": st.session_state.workweek, "holidays": [d for d in sorted(list(st.session_state.holidays))]},
        "scenarios": st.session_state.scenarios,  # keep as-is for v6.1
    }
    return json.dumps(data, indent=2, default=str)

def import_json(text: str):
    reset_all()
    data = json.loads(text)
    engs = data.get("engineers", [])
    for e in engs: e["time_off"] = set(e.get("time_off", []))
    st.session_state.engineers = engs
    st.session_state.tasks = data.get("tasks", [])
    st.session_state.workweek = data.get("settings", {}).get("workweek", st.session_state.workweek)
    st.session_state.holidays = set(data.get("settings", {}).get("holidays", []))
    st.session_state.allocations = parse_alloc(data.get("allocations", {}))
    st.session_state.baseline_allocations = parse_alloc(data.get("baseline_allocations", {}))
    st.session_state.scenarios = data.get("scenarios", [])

def next_id(items):
    return 1 if not items else max(x["id"] for x in items) + 1

def engineer_effective_rating(engineer, required_skills):
    ratings = []
    skills = engineer.get("skills", {})
    for s in required_skills:
        ratings.append(skills.get(s, 1))
    return min(ratings) if ratings else 1

def is_global_workday(d: date):
    return st.session_state.workweek.get(d.weekday(), False) and (d.isoformat() not in st.session_state.holidays)

def is_engineer_workday(eng, d: date):
    return is_global_workday(d) and (d.isoformat() not in eng.get("time_off", set())) and (eng.get("capacity_pct",100) > 0)

def next_work_start(eng, start_dt: datetime) -> datetime:
    cur = start_dt
    while not is_engineer_workday(eng, cur.date()):
        cur += timedelta(days=1)
    return cur

def add_workdays_for_engineer(eng, start_dt: datetime, eff_days: float) -> datetime:
    cur = next_work_start(eng, start_dt)
    remaining = eff_days
    per_day = max(0.01, eng.get("capacity_pct",100)/100.0)
    while remaining > 1e-9:
        if is_engineer_workday(eng, cur.date()):
            remaining -= per_day
        cur += timedelta(days=1)
    return cur

def overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])

def find_earliest_slot(eng, busy_list, earliest: datetime, eff_days: float):
    start = next_work_start(eng, earliest)
    end = add_workdays_for_engineer(eng, start, eff_days)
    for b_start, b_end in sorted(busy_list):
        if overlap((start, end), (b_start, b_end)):
            start = next_work_start(eng, b_end)
            end = add_workdays_for_engineer(eng, start, eff_days)
    return start

def topological_order(tasks):
    id_to_task = {t["id"]: t for t in tasks}
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for t in tasks:
        pred = t.get("predecessor_id")
        if pred is not None and pred in id_to_task:
            adj[pred].append(t["id"])
            indeg[t["id"]] += 1
        else:
            indeg[t["id"]] += 0
    q = deque([tid for tid in id_to_task if indeg[tid] == 0])
    order = []
    visited = 0
    while q:
        u = q.popleft()
        order.append(u)
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    if visited != len(tasks):
        st.warning("Cyclic dependency detected or invalid predecessor references. Order may be incomplete.")
    return order, adj

def plan_schedule_on(engineers, tasks):
    allocations = {}
    calendars = defaultdict(list)
    id_to_task = {t["id"]: t for t in tasks}
    order, _ = topological_order(tasks)
    for tid in order:
        t = id_to_task[tid]
        req_skills = t.get("req_skills", [])
        if not req_skills: continue
        candidates = []
        for eng in engineers:
            eff = engineer_effective_rating(eng, req_skills)
            if eff >= 2 and eng.get("capacity_pct",100) > 0:
                candidates.append((eng, eff))
        if not candidates: continue
        earliest = datetime.combine(to_date(t["start_date"]), datetime.min.time())
        pred = t.get("predecessor_id")
        if pred is not None and pred in allocations:
            earliest = max(earliest, allocations[pred]["end"])
        est_md = float(t["est_duration_md"])
        best_choice, best_finish = None, None
        for eng, eff in candidates:
            factor = SKILL_FACTORS.get(eff, None)
            if factor is None: continue
            base_days = est_md * factor
            cap = max(1, eng.get("capacity_pct",100))
            eff_days = base_days * (100.0 / cap)
            start_time = find_earliest_slot(eng, calendars[eng["id"]], earliest, eff_days)
            end_time = add_workdays_for_engineer(eng, start_time, eff_days)
            if best_finish is None or end_time < best_finish:
                best_finish = end_time
                best_choice = (eng, eff, start_time, end_time, base_days, factor, cap, eff_days)
        if best_choice is None: continue
        eng, eff, start_time, end_time, base_days, factor, cap, eff_days = best_choice
        calendars[eng["id"]].append((start_time, end_time))
        calendars[eng["id"]].sort()
        allocations[tid] = {
            "engineer_id": eng["id"],
            "start": start_time,
            "end": end_time,
            "duration_days": int(round(base_days)),
            "factor": factor,
            "capacity_pct": cap,
            "eff_duration_days": eff_days
        }
    return allocations

def plan_schedule():
    st.session_state.allocations = plan_schedule_on(st.session_state.engineers, st.session_state.tasks)
    st.session_state.dirty = True

def compute_critical_path_and_slack(tasks, allocations):
    if not tasks or not allocations: return set(), {}
    id_to_task = {t["id"]: t for t in tasks}
    order, adj = topological_order(tasks)
    duration = {tid: max(1, int(round(allocations[tid]["eff_duration_days"]))) if tid in allocations else 0 for tid in id_to_task.keys()}
    ES = {tid: 0 for tid in id_to_task}; EF = {tid: duration[tid] for tid in id_to_task}
    for u in order:
        for v in adj[u]:
            ES[v] = max(ES[v], EF[u])
            EF[v] = ES[v] + duration[v]
    project_length = max(EF.values()) if EF else 0
    rev_order = order[::-1]
    LF = {tid: project_length for tid in id_to_task}
    LS = {tid: LF[tid] - duration[tid] for tid in id_to_task}
    for u in rev_order:
        if adj[u]:
            LF[u] = min(LS[v] for v in adj[u])
            LS[u] = LF[u] - duration[u]
    slack = {tid: LS[tid] - ES[tid] for tid in id_to_task}
    cp = {tid for tid, s in slack.items() if s == 0}
    return cp, slack

def kpis(tasks, allocations):
    if not allocations:
        return {"ProjectFinish": None, "LateTasks": 0, "TotalEffDays": 0, "CP_Length": 0}
    finish = max(a["end"].date() for a in allocations.values())
    id_to_task = {t["id"]: t for t in tasks}
    late = sum(1 for tid, a in allocations.items() if a["end"].date() > to_date(id_to_task[tid]["due_date"]))
    total_eff = int(round(sum(a["eff_duration_days"] for a in allocations.values())))
    cp, _ = compute_critical_path_and_slack(tasks, allocations)
    return {"ProjectFinish": finish, "LateTasks": late, "TotalEffDays": total_eff, "CP_Length": len(cp)}

def render_gantt(tasks, allocations, title="Gantt"):
    if not tasks or not allocations:
        st.info("No schedule to display."); return
    cp, slack = compute_critical_path_and_slack(tasks, allocations)
    eng_map = {e["id"]: e["name"] for e in st.session_state.engineers}
    rows = []
    for t in tasks:
        tid = t["id"]
        if tid not in allocations: continue
        a = allocations[tid]
        due = datetime.combine(to_date(t["due_date"]), datetime.min.time())
        late = a["end"].date() > to_date(t["due_date"])
        rows.append({
            "Task": t["name"],
            "Engineer": eng_map.get(a["engineer_id"], "Unassigned"),
            "Start": a["start"], "Finish": a["end"],
            "DueDate": due, "Late": "LATE" if late else "On track",
            "CriticalPath": "On CP" if t["id"] in cp else "Off CP",
            "SlackDays": slack.get(t["id"], 0),
            "Capacity%": a.get("capacity_pct", 100), "SkillFactor": a.get("factor", 1.0),
        })
    df = pd.DataFrame(rows)
    df["Status"] = df["Late"].mask(df["Late"]=="On track", df["CriticalPath"])
    st.subheader(title)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Status",
                      hover_data=["Engineer","SlackDays","Capacity%","SkillFactor"])
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["Task","Engineer","Start","Finish","DueDate","Late","CriticalPath","SlackDays","Capacity%","SkillFactor"]])

# --- Full-featured editors (from v6) ---
def engineer_editor():
    st.header("Engineer Database")
    engs = st.session_state.engineers
    with st.expander("➕ Add engineer", expanded=False):
        name = st.text_input("Name", key="new_eng_name")
        skills_text = st.text_area("Skills and ratings (e.g., 'C++:5' per line)", key="new_eng_skills")
        capacity = st.slider("Capacity (%)", 0, 100, 100, 5, key="new_eng_capacity")
        time_off_text = st.text_area("Personal time off (ISO dates, one per line)", key="new_eng_timeoff")
        if st.button("Add engineer"):
            if not name.strip():
                st.warning("Please enter a name.")
            else:
                eng = {"id": next_id(engs), "name": name.strip(), "skills": {}, "capacity_pct": capacity, "time_off": set()}
                for line in skills_text.splitlines():
                    if ":" in line:
                        skill, rating = line.split(":", 1)
                        try:
                            r = max(1, min(5, int(rating.strip()))); eng["skills"][skill.strip()] = r
                        except: pass
                for d in [t.strip() for t in time_off_text.splitlines() if t.strip()]:
                    eng["time_off"].add(d)
                engs.append(eng); st.session_state.dirty = True; st.success(f"Engineer '{name}' added.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬇️ Export engineers CSV"):
            df = pd.DataFrame([{"id":e["id"], "name":e["name"], "capacity_pct": e.get("capacity_pct",100),
                                "time_off": ",".join(sorted(list(e.get("time_off",set())))),
                                **{f"skill:{k}":v for k,v in e.get("skills",{}).items()}} for e in engs])
            st.download_button("Download", data=df.to_csv(index=False), mime="text/csv", file_name="engineers.csv")
    with c2:
        up = st.file_uploader("⬆️ Import engineers CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up); new_list = []
            for _, r in df.iterrows():
                skills = {k.split("skill:",1)[1]: int(r[k]) for k in df.columns if k.startswith("skill:") and not pd.isna(r[k])}
                time_off = set([] if pd.isna(r.get("time_off","")) else str(r.get("time_off","")).split(","))
                cap = 100 if pd.isna(r.get("capacity_pct")) else int(r.get("capacity_pct"))
                new_list.append({"id": int(r["id"]), "name": r["name"], "skills": skills, "capacity_pct": cap,
                                 "time_off": set(t.strip() for t in time_off if t.strip())})
            st.session_state.engineers = new_list; st.success("Engineers imported.")

    for eng in engs:
        with st.expander(f"👤 {eng['name']}", expanded=False):
            eng["name"] = st.text_input("Name", value=eng["name"], key=f"eng_name_{eng['id']}")
            eng["capacity_pct"] = st.slider("Capacity (%)", 0, 100, int(eng.get("capacity_pct",100)), 5, key=f"cap_{eng['id']}")
            to_delete = []
            st.write("Skills")
            for skill, rating in list(eng.get("skills", {}).items()):
                c1, c2, c3 = st.columns([2,1,1])
                with c1:
                    new_skill = st.text_input("Skill", value=skill, key=f"skill_{eng['id']}_{skill}")
                with c2:
                    new_rating = st.slider("Rating", 1, 5, int(rating), 1, key=f"rate_{eng['id']}_{skill}")
                with c3:
                    if st.button("Delete", key=f"del_{eng['id']}_{skill}"): to_delete.append(skill)
                if new_skill != skill:
                    eng["skills"].pop(skill); eng["skills"][new_skill] = new_rating
                else:
                    eng["skills"][skill] = new_rating
            for s in to_delete: eng["skills"].pop(s, None)

            c1, c2 = st.columns([2,1])
            with c1: add_skill = st.text_input("Add Skill", key=f"add_skill_{eng['id']}")
            with c2: add_rating = st.slider("Rating", 1, 5, 3, 1, key=f"add_rate_{eng['id']}")
            if st.button("Add skill", key=f"add_btn_{eng['id']}"):
                if add_skill.strip():
                    eng["skills"][add_skill.strip()] = add_rating; st.session_state.dirty = True; st.success(f"Added {add_skill}")

            st.write("Personal time off (ISO dates, one per line)")
            timeoff_text = "\n".join(sorted(list(eng.get("time_off", set()))))
            new_text = st.text_area("Time off", value=timeoff_text, height=120, key=f"timeoff_{eng['id']}")
            if st.button("Save time off", key=f"save_timeoff_{eng['id']}"):
                eng["time_off"] = set(t.strip() for t in new_text.splitlines() if t.strip()); st.success("Saved.")

def task_editor():
    st.header("Tasks")
    with st.expander("➕ Add task", expanded=False):
        name = st.text_input("Task name", key="new_task_name")
        desc = st.text_area("Description", key="new_task_desc")
        start_date = st.date_input("Planned start date", key="new_task_start", value=datetime.now().date())
        est = st.number_input("Estimated duration (man-days)", min_value=1.0, step=0.5, value=5.0, key="new_task_est")
        req_skills = st.text_input("Required skills (comma-separated, up to 3)", key="new_task_skills")
        due_date = st.date_input("Target completion date", key="new_task_due", value=datetime.now().date() + timedelta(days=10))
        predecessor = st.selectbox("Predecessor (optional)", options=["None"] + [f'{t["id"]}: {t["name"]}' for t in st.session_state.tasks], key="new_task_pred")
        pred_id = None if predecessor=="None" else int(predecessor.split(":")[0])
        if st.button("Add task"):
            skills = [s.strip() for s in req_skills.split(",") if s.strip()]
            if not (1 <= len(skills) <= 3):
                st.warning("Please provide 1 to 3 required skills.")
            else:
                tid = next_id(st.session_state.tasks)
                st.session_state.tasks.append({"id":tid,"name":name.strip(),"desc":desc.strip(),
                                               "start_date": start_date.isoformat(),"est_duration_md": float(est),
                                               "req_skills": skills,"due_date": due_date.isoformat(),
                                               "predecessor_id": pred_id})
                st.session_state.dirty = True; st.success(f"Task '{name}' added.")

    if not st.session_state.tasks: st.info("No tasks yet.")
    else:
        for t in st.session_state.tasks:
            with st.expander(f"🧩 {t['id']}: {t['name']}", expanded=False):
                t["name"] = st.text_input("Name", value=t["name"], key=f"t_name_{t['id']}")
                t["desc"] = st.text_area("Description", value=t["desc"], key=f"t_desc_{t['id']}")
                t["start_date"] = st.date_input("Start Date", value=to_date(t["start_date"]), key=f"t_start_{t['id']}").isoformat()
                t["est_duration_md"] = float(st.number_input("Estimated duration (man-days)", 1.0, step=0.5, value=float(t["est_duration_md"]), key=f"t_est_{t['id']}"))
                skills_str = ", ".join(t.get("req_skills", []))
                t["req_skills"] = [s.strip() for s in st.text_input("Required skills (comma-separated)", value=skills_str, key=f"t_skills_{t['id']}").split(",") if s.strip()]
                t["due_date"] = st.date_input("Target completion date", value=to_date(t["due_date"]), key=f"t_due_{t['id']}").isoformat()
                pred_opt = ["None"] + [f'{x["id"]}: {x["name"]}' for x in st.session_state.tasks if x["id"] != t["id"]]
                cur_pred = "None" if not t.get("predecessor_id") else next((o for o in pred_opt if o.startswith(f'{t["predecessor_id"]}:')), "None")
                sel = st.selectbox("Predecessor", options=pred_opt, index=pred_opt.index(cur_pred) if cur_pred in pred_opt else 0, key=f"t_pred_{t['id']}")
                t["predecessor_id"] = None if sel == "None" else int(sel.split(":")[0])
                if st.button("Delete task", key=f"del_task_{t['id']}"):
                    st.session_state.tasks = [x for x in st.session_state.tasks if x["id"] != t["id"]]
                    st.session_state.allocations.pop(t["id"], None); st.session_state.dirty = True; st.rerun()

def settings_editor():
    st.header("Settings")
    st.caption("Workweek and holidays.")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    cols = st.columns(7)
    for i, c in enumerate(cols):
        with c: st.session_state.workweek[i] = st.checkbox(days[i], value=st.session_state.workweek.get(i, False))
    st.write("Global Holidays (ISO date, one per line)")
    holidays_text = "\n".join(sorted(list(st.session_state.holidays)))
    new_text = st.text_area("Holidays", value=holidays_text, height=120)
    if st.button("Save holidays"):
        st.session_state.holidays = set(t.strip() for t in new_text.splitlines() if t.strip()); st.success("Saved.")
    st.subheader("Country holiday library")
    country = st.selectbox("Select", ["—"] + sorted(COUNTRY_HOLIDAYS.keys()))
    if st.button("Load library"):
        if country in COUNTRY_HOLIDAYS:
            st.session_state.holidays.update(COUNTRY_HOLIDAYS[country]); st.success(f"Loaded {country} holidays.")

def planning_tab():
    st.header("Plan & Gantt")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Auto-plan"): plan_schedule(); st.success("Planning complete.")
    with c2:
        if st.button("Clear allocations"): st.session_state.allocations = {}; st.session_state.dirty = True; st.info("Allocations cleared.")
    with c3:
        if st.button("Set Baseline (snapshot)"):
            if st.session_state.allocations:
                st.session_state.baseline_allocations = json.loads(json.dumps(st.session_state.allocations, default=str))
                for k, v in st.session_state.baseline_allocations.items():
                    v["start"] = datetime.fromisoformat(v["start"]); v["end"] = datetime.fromisoformat(v["end"])
                st.success("Baseline captured from current plan.")
            else: st.warning("Create a plan first.")
    with c4:
        if st.button("Show Variance Table"):
            rows = compute_variance_rows()
            if rows:
                dfv = pd.DataFrame(rows); st.dataframe(dfv)
                st.download_button("⬇️ Download variance CSV", data=dfv.to_csv(index=False), mime="text/csv", file_name="variance.csv")
            else: st.info("No baseline set yet.")
    render_gantt(st.session_state.tasks, st.session_state.allocations, title="Current Plan")

def compute_variance_rows():
    base = st.session_state.baseline_allocations; curr = st.session_state.allocations; rows = []
    if not base: return rows
    for tid_str, b in base.items():
        tid = int(tid_str); 
        if tid not in curr: continue
        c = curr[tid]
        rows.append({"TaskID": tid, "BaselineStart": b["start"].date(), "BaselineFinish": b["end"].date(),
                     "CurrentStart": c["start"].date(), "CurrentFinish": c["end"].date(),
                     "StartSlipDays": (c["start"].date() - b["start"].date()).days,
                     "FinishSlipDays": (c["end"].date() - b["end"].date()).days,
                     "DurationΔDays": int(round(c["eff_duration_days"] - b.get("eff_duration_days", c["eff_duration_days"])))})
    return rows

def weekly_utilization_df(start_date=None, weeks=8):
    if not st.session_state.allocations or not st.session_state.engineers: return pd.DataFrame()
    allocs = st.session_state.allocations; eng_map = {e["id"]: e for e in st.session_state.engineers}
    if start_date is None: start_date = min(a["start"].date() for a in allocs.values())
    start_date = start_date - timedelta(days=start_date.weekday())
    weeks_range = [start_date + timedelta(weeks=w) for w in range(weeks)]
    data = []
    for eng_id, eng in eng_map.items():
        for week_start in weeks_range:
            week_end = week_start + timedelta(days=7)
            cap = max(0.01, eng.get("capacity_pct",100)/100.0)
            eff_days = 0.0
            for tid, a in allocs.items():
                if a["engineer_id"] != eng_id: continue
                d = a["start"].date()
                while d < a["end"].date():
                    if week_start <= d < week_end and is_engineer_workday(eng, d): eff_days += cap
                    d += timedelta(days=1)
            week_workdays = sum(1 for i in range(7) if is_engineer_workday(eng, week_start + timedelta(days=i)))
            max_eff = week_workdays * cap if week_workdays > 0 else 0.01
            utilization = eff_days / max_eff if max_eff > 0 else 0.0
            data.append({"Engineer": eng["name"], "WeekStart": week_start, "Utilization": utilization})
    return pd.DataFrame(data)

def weekly_view_tab():
    st.header("Weekly Utilization")
    weeks = st.slider("Weeks forward", 4, 26, 12, 1)
    df = weekly_utilization_df(weeks=weeks)
    if df.empty: st.info("Create a plan to see utilization."); return
    pivot = df.pivot(index="WeekStart", columns="Engineer", values="Utilization").fillna(0.0).reset_index()
    st.dataframe((pivot.set_index("WeekStart")*100).round(1))
    st.download_button("⬇️ Download weekly utilization CSV", data=pivot.to_csv(index=False), mime="text/csv", file_name="weekly_utilization.csv")
    for eng in [c for c in pivot.columns if c != "WeekStart"]:
        fig = px.line(pivot, x="WeekStart", y=eng, title=f"Utilization for {eng}"); st.plotly_chart(fig, use_container_width=True)

# ---- Home with extra options ----
def home_screen():
    st.title("🛠️ SkillGantt v6.1 — Skill-based Gantt Planner")
    st.caption("Now with Home shortcuts to edit Engineers and Tasks directly.")
    choice = st.radio("Choose an option:", [
        "Load a previously saved Gantt",
        "Create a new one",
        "Modify the database of engineers",
        "Modify the task list",
        "Modify engineers & tasks together"
    ])

    if choice == "Load a previously saved Gantt":
        st.subheader("Load")
        uploaded = st.file_uploader("Upload a .json export", type=["json"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8"); import_json(content); st.success("Project loaded.")
            st.info("Use the tabs above to continue (👥 Engineers / 🧩 Tasks / 🗓️ Plan & Gantt).")

    elif choice == "Create a new one":
        st.subheader("New Project")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start fresh"): reset_all(); st.success("New empty project created.")
        with c2:
            if st.button("Load sample data"): load_sample(); st.success("Sample data loaded.")

    elif choice == "Modify the database of engineers":
        st.subheader("Engineers")
        engineer_editor()

    elif choice == "Modify the task list":
        st.subheader("Tasks")
        task_editor()

    else:  # both
        st.subheader("Engineers & Tasks")
        c1, c2 = st.columns(2)
        with c1: engineer_editor()
        with c2: task_editor()

def what_if_tab():
    st.header("🔮 What‑If (quick)")
    st.caption("This streamlined What‑If tab keeps the full editors elsewhere. For advanced scenarios use SkillGantt v6 full build.")
    # Minimal what-if: global capacity & duration multipliers
    cap_mult = st.slider("Global capacity multiplier", 50, 150, 100, 5)
    dur_mult = st.slider("Global task duration multiplier", 50, 200, 100, 5)
    engineers = copy.deepcopy(st.session_state.engineers)
    tasks = copy.deepcopy(st.session_state.tasks)
    for e in engineers:
        e["capacity_pct"] = int(round(e.get("capacity_pct",100) * cap_mult / 100.0))
    for t in tasks:
        t["est_duration_md"] = float(t["est_duration_md"]) * (dur_mult/100.0)
    if st.button("Run what‑if plan"):
        alloc = plan_schedule_on(engineers, tasks)
        render_gantt(tasks, alloc, title="What‑If Plan")

def load_sample():
    reset_all()
    st.session_state.engineers = [
        {"id":1,"name":"Alice","capacity_pct":100,"time_off":set(),"skills":{"C++":5,"Python":4,"Perception":3}},
        {"id":2,"name":"Bob","capacity_pct":80,"time_off":set(["2025-11-24","2025-11-25"]), "skills":{"C++":3,"Python":5,"Planning":4}},
        {"id":3,"name":"Carla","capacity_pct":60,"time_off":set(),"skills":{"Controls":4,"Planning":2,"Python":3}},
    ]
    today = datetime.now().date()
    st.session_state.tasks = [
        {"id":1,"name":"Perception pipeline","desc":"Implement detector","start_date": (today).isoformat(),"est_duration_md":5.0,"req_skills":["Python","Perception"],"due_date": (today+ timedelta(days=14)).isoformat(),"predecessor_id": None},
        {"id":2,"name":"Planner","desc":"MPC improvements","start_date": (today+ timedelta(days=1)).isoformat(),"est_duration_md":6.0,"req_skills":["C++","Planning"],"due_date": (today+ timedelta(days=20)).isoformat(),"predecessor_id": 1},
        {"id":3,"name":"Controls","desc":"Tune controllers","start_date": (today+ timedelta(days=2)).isoformat(),"est_duration_md":4.0,"req_skills":["Controls"],"due_date": (today+ timedelta(days=15)).isoformat(),"predecessor_id": 2},
    ]

# ---- Main ----
load_state()
tabs = st.tabs(["🏠 Home", "👥 Engineers", "🧩 Tasks", "⚙️ Settings", "🗓️ Plan & Gantt", "📊 Weekly Utilization", "🔮 What‑If", "📦 Export / Import"])
with tabs[0]: home_screen()
with tabs[1]: engineer_editor()
with tabs[2]: task_editor()
with tabs[3]: settings_editor()
with tabs[4]: planning_tab()
with tabs[5]: weekly_view_tab()
with tabs[6]: what_if_tab()
with tabs[7]:
    st.header("Export / Import")
    st.download_button("⬇️ Download project (.json)", data=export_json(), mime="application/json", file_name="skillgantt_project_v6_1.json")
    st.write("---")
    st.subheader("Import")
    up = st.file_uploader("Upload a .json export to replace current session", type=["json"], key="imp2")
    if up is not None:
        content = up.read().decode("utf-8"); import_json(content); st.success("Project imported.")
