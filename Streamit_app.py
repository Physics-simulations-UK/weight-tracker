import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import List, Literal

import pandas as pd
import streamlit as st
from google import genai
from pydantic import BaseModel, Field


# =========================================================
# PAGE + STORAGE
# =========================================================

st.set_page_config(
    page_title="Weight & Nutrition Tracker",
    page_icon="⚖️",
    layout="wide",
)

DATA_FILE = Path("tracker_data.json")

DEFAULT_DATA = {
    "weights": [],
    "foods": [],
    "steps": [],
    "activities": [],
    "saved_foods": [],
    "settings": {
        "bmr": 1744,
        "step_factor": 0.045,
        "gemini_model": "gemini-3.1-flash-lite",
    },
}

MEALS = ["Breakfast", "Lunch", "Dinner", "Snacks"]

ACTIVITY_PRESETS = {
    "Golf – 9 holes (cart)": {"type": "fixed", "calories": 300},
    "Golf – 18 holes (cart)": {"type": "fixed", "calories": 600},
    "Golf – 9 holes (walking)": {"type": "fixed", "calories": 450},
    "Golf – 18 holes (walking)": {"type": "fixed", "calories": 900},
    "Strength training": {"type": "met", "met": 3.5},
    "Bodyweight exercise": {"type": "met", "met": 3.0},
    "Kettlebell swings": {"type": "met", "met": 9.8},
    "Elliptical trainer": {"type": "met", "met": 5.0},
    "Rowing machine": {"type": "met", "met": 5.0},
    "Stationary cycling": {"type": "met", "met": 6.8},
    "Spin class": {"type": "met", "met": 9.0},
    "Swimming – leisurely": {"type": "met", "met": 6.0},
    "Swimming laps – recreational": {"type": "met", "met": 5.8},
    "Yoga": {"type": "met", "met": 2.3},
    "Custom activity": {"type": "custom"},
}


def fresh_default_data():
    return json.loads(json.dumps(DEFAULT_DATA))


def load_data():
    if not DATA_FILE.exists():
        return fresh_default_data()

    try:
        loaded = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fresh_default_data()

    data = fresh_default_data()

    for key in ["weights", "foods", "steps", "activities", "saved_foods"]:
        if isinstance(loaded.get(key), list):
            data[key] = loaded[key]

    if isinstance(loaded.get("settings"), dict):
        data["settings"].update(loaded["settings"])

    return data


def save_data():
    DATA_FILE.write_text(
        json.dumps(st.session_state.data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if "data" not in st.session_state:
    st.session_state.data = load_data()

data = st.session_state.data


# =========================================================
# HELPERS
# =========================================================

def iso(d):
    return d.isoformat() if isinstance(d, date) else str(d)


def as_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def new_id():
    return str(uuid.uuid4())


def sort_records(records):
    return sorted(records, key=lambda x: x["date"])


def upsert_by_date(collection_name, record):
    collection = data[collection_name]
    for index, item in enumerate(collection):
        if item["date"] == record["date"]:
            collection[index] = record
            save_data()
            return
    collection.append(record)
    save_data()


def get_weight_record(day):
    day = iso(day)
    return next((w for w in data["weights"] if w["date"] == day), None)


def get_weight_for_date(day):
    """Most recent weigh-in on or before the selected day."""
    day = iso(day)
    available = [
        w for w in sort_records(data["weights"])
        if w["date"] <= day
    ]
    if available:
        return float(available[-1]["weight"])
    if data["weights"]:
        return float(sort_records(data["weights"])[0]["weight"])
    return None


def rolling_average_for_weight_date(day, days):
    ordered = sort_records(data["weights"])
    idx = next((i for i, w in enumerate(ordered) if w["date"] == iso(day)), None)
    if idx is None:
        return None

    start = max(0, idx - days + 1)
    values = [float(w["weight"]) for w in ordered[start:idx + 1]]
    return sum(values) / len(values) if values else None


def get_steps(day):
    day = iso(day)
    record = next((x for x in data["steps"] if x["date"] == day), None)
    return int(record["steps"]) if record else None


def get_foods(day, meal=None):
    day = iso(day)
    rows = [f for f in data["foods"] if f["date"] == day]
    if meal:
        rows = [f for f in rows if f["meal"] == meal]
    return rows


def day_calories(day):
    return sum(float(f.get("calories", 0)) for f in get_foods(day))


def day_protein(day):
    return sum(float(f.get("protein", 0)) for f in get_foods(day))


def get_activities(day):
    day = iso(day)
    return [a for a in data["activities"] if a["date"] == day]


def day_activity_calories(day):
    return sum(float(a.get("calories", 0)) for a in get_activities(day))


def tdee(day):
    steps = get_steps(day)
    if steps is None:
        return None

    settings = data["settings"]
    return (
        float(settings["bmr"])
        + steps * float(settings["step_factor"])
        + day_activity_calories(day)
    )


def daily_deficit(day):
    value = tdee(day)
    if value is None:
        return None
    return value - day_calories(day)


def all_dates():
    dates = {
        x["date"]
        for group in ["weights", "foods", "steps", "activities"]
        for x in data[group]
    }
    return sorted(dates)


def cumulative_deficit_map():
    total = 0.0
    result = {}
    for day in all_dates():
        deficit = daily_deficit(day)
        if deficit is not None:
            total += deficit
        result[day] = total
    return result


def calculate_met_calories(met, weight_kg, minutes):
    """
    NET exercise calories.
    Subtract 1 MET because resting expenditure is already represented by BMR.
    """
    net_met = max(float(met) - 1.0, 0.0)
    return net_met * float(weight_kg) * (float(minutes) / 60.0)


def add_food_record(day, meal, name, calories, protein, quantity="",
                    source="manual", confidence="", assumption=""):
    data["foods"].append({
        "id": new_id(),
        "date": iso(day),
        "meal": meal,
        "name": str(name).strip(),
        "quantity": str(quantity).strip(),
        "calories": round(float(calories), 1),
        "protein": round(float(protein), 1),
        "source": source,
        "confidence": confidence,
        "assumption": assumption,
    })
    save_data()


def delete_by_id(collection_name, record_id):
    data[collection_name] = [
        x for x in data[collection_name]
        if x.get("id") != record_id
    ]
    save_data()


def delete_day(day):
    day = iso(day)
    for collection in ["weights", "foods", "steps", "activities"]:
        data[collection] = [
            x for x in data[collection]
            if x["date"] != day
        ]
    save_data()


def current_weight():
    if not data["weights"]:
        return None
    return float(sort_records(data["weights"])[-1]["weight"])


# =========================================================
# GEMINI FOOD ANALYSIS
# =========================================================

class FoodItem(BaseModel):
    name: str = Field(description="Clear food name")
    quantity: str = Field(
        description="Quantity or portion as described or reasonably inferred, including units"
    )
    calories: float = Field(
        ge=0, description="Estimated or calculated calories for this specific quantity"
    )
    protein: float = Field(
        ge=0, description="Estimated or calculated protein grams for this specific quantity"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this food estimate"
    )
    source_type: Literal[
        "user_provided", "saved_food", "web_researched", "estimated"
    ] = Field(
        description="Main basis for the nutrition value"
    )
    assumption: str = Field(
        description="Short explanation of assumptions or calculation; empty string if none"
    )


class FoodAnalysis(BaseModel):
    items: List[FoodItem]
    total_calories: float = Field(ge=0)
    total_protein: float = Field(ge=0)
    summary: str = Field(
        description="One short sentence summarising important assumptions"
    )


@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def get_saved_food_catalogue():
    return [
        {
            "name": f["name"],
            "basis": f.get("basis", "portion"),
            "quantity": f.get("quantity", ""),
            "calories": f.get("calories", 0),
            "protein": f.get("protein", 0),
            "calories_per_100g": f.get("calories_per_100g"),
            "protein_per_100g": f.get("protein_per_100g"),
        }
        for f in data["saved_foods"]
    ]


def extract_grounding_sources(response):
    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        chunks = getattr(metadata, "grounding_chunks", None) or []
        seen = set()
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            title = getattr(web, "title", "") or "Web source"
            uri = getattr(web, "uri", "") or ""
            key = (title, uri)
            if key not in seen:
                seen.add(key)
                sources.append({"title": title, "uri": uri})
    except Exception:
        pass
    return sources


def analyse_food_with_gemini(description):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from Streamlit secrets."
        )

    client = get_gemini_client(api_key)
    model = data["settings"].get("gemini_model", "gemini-3.1-flash-lite")
    saved_catalogue = get_saved_food_catalogue()

    prompt = f"""
You are the nutrition-analysis engine for a personal weight-loss food tracker.

Analyse the user's food description into separate food items and calculate calories
and protein for the quantity actually eaten.

USER FOOD DESCRIPTION:
{description}

SAVED FOOD CATALOGUE:
{json.dumps(saved_catalogue, ensure_ascii=False)}

RULES:
1. Exact information supplied by the user always wins over estimates.
   Example: "200g hash browns at 159 kcal per 100g" must calculate exactly from 159 kcal/100g.
2. If an item clearly matches a saved food, use the saved food value rather than estimating it.
3. When no exact or saved value exists, use Google Search when useful to find a reliable nutrition
   value, prioritising manufacturers, supermarkets, restaurant nutrition pages, USDA/government
   databases, or other reputable nutrition sources.
4. If a branded product is named, actively prefer the manufacturer's or retailer's nutrition data.
5. If no reliable specific value can be found, make a sensible conservative estimate.
6. Do not invent false precision. Round calories sensibly and protein to about 0.1 g where useful.
7. If portion size is unclear, state the assumption and reduce confidence.
8. Calories and protein must correspond to the stated quantity, not per-100g unless the quantity itself is 100g.
9. The totals must equal the sum of the returned items.
10. Use source_type:
   - user_provided for calculations based on nutrition values in the user's message
   - saved_food for values taken from the saved catalogue
   - web_researched when Google Search was used
   - estimated when neither an exact, saved nor web-specific value was available.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "tools": [{"google_search": {}}],
            "response_mime_type": "application/json",
            "response_schema": FoodAnalysis,
        },
    )

    result = FoodAnalysis.model_validate_json(response.text)
    return result, extract_grounding_sources(response)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚖️ Tracker")

with st.sidebar.expander("Calculation settings", expanded=False):
    bmr = st.number_input(
        "BMR (kcal)",
        min_value=500,
        max_value=4000,
        value=int(data["settings"]["bmr"]),
        step=1,
    )
    step_factor = st.number_input(
        "Calories per step",
        min_value=0.001,
        max_value=0.200,
        value=float(data["settings"]["step_factor"]),
        step=0.001,
        format="%.3f",
    )
    gemini_model = st.text_input(
        "Gemini model",
        value=data["settings"].get("gemini_model", "gemini-3.1-flash-lite"),
    )

    if st.button("Save settings", use_container_width=True):
        data["settings"]["bmr"] = int(bmr)
        data["settings"]["step_factor"] = float(step_factor)
        data["settings"]["gemini_model"] = gemini_model.strip() or "gemini-3.1-flash-lite"
        save_data()
        st.success("Settings saved.")

st.sidebar.caption(
    "TDEE = BMR + step calories + additional activity calories"
)


# =========================================================
# TITLE
# =========================================================

st.title("Weight & Nutrition Tracker")
st.caption(
    "Weight trends, intelligent food analysis, activity and calorie balance"
)

tabs = st.tabs([
    "📊 Dashboard",
    "📝 Log day",
    "✨ AI food",
    "⭐ Saved foods",
    "💾 Data",
])


# =========================================================
# DASHBOARD
# =========================================================

with tabs[0]:
    weights_sorted = sort_records(data["weights"])
    latest_weight = current_weight()

    if weights_sorted:
        last_day = weights_sorted[-1]["date"]
        avg3 = rolling_average_for_weight_date(last_day, 3)
        avg7 = rolling_average_for_weight_date(last_day, 7)
        total_change = latest_weight - float(weights_sorted[0]["weight"])
    else:
        avg3 = avg7 = total_change = None

    today_iso = date.today().isoformat()
    today_def = daily_deficit(today_iso)
    cumulative = cumulative_deficit_map()
    cumulative_total = cumulative.get(all_dates()[-1], 0) if all_dates() else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Latest weight",
        f"{latest_weight:.2f} kg" if latest_weight is not None else "—",
    )
    c2.metric(
        "3-day average",
        f"{avg3:.2f} kg" if avg3 is not None else "—",
    )
    c3.metric(
        "7-day average",
        f"{avg7:.2f} kg" if avg7 is not None else "—",
    )
    c4.metric(
        "Total change",
        f"{total_change:+.2f} kg" if total_change is not None else "—",
    )
    c5.metric(
        "Today's deficit",
        f"{today_def:+.0f} kcal" if today_def is not None else "—",
    )
    c6.metric(
        "Cumulative deficit",
        f"{cumulative_total:+,.0f} kcal",
    )

    st.subheader("Weight trend")

    if weights_sorted:
        rows = []
        for i, w in enumerate(weights_sorted):
            values3 = [
                float(x["weight"])
                for x in weights_sorted[max(0, i - 2):i + 1]
            ]
            values7 = [
                float(x["weight"])
                for x in weights_sorted[max(0, i - 6):i + 1]
            ]
            rows.append({
                "Date": as_date(w["date"]),
                "Daily weight": float(w["weight"]),
                "3-day average": sum(values3) / len(values3),
                "7-day average": sum(values7) / len(values7),
            })

        chart_df = pd.DataFrame(rows).set_index("Date")
        st.line_chart(
            chart_df[["Daily weight", "3-day average", "7-day average"]],
            height=400,
        )
    else:
        st.info("Add your first weigh-in to start the weight graph.")

    st.subheader("Daily tracking")

    cumulative = cumulative_deficit_map()
    table_rows = []

    for day in reversed(all_dates()):
        wr = get_weight_record(day)
        activity_names = ", ".join(a["name"] for a in get_activities(day))

        meal_text = {}
        for meal in MEALS:
            items = get_foods(day, meal)
            meal_text[meal] = "; ".join(
                f'{f["name"]}'
                + (f' ({f.get("quantity", "")})' if f.get("quantity") else "")
                for f in items
            )

        table_rows.append({
            "Date": day,
            "Weight": round(float(wr["weight"]), 2) if wr else None,
            "3-day avg": round(rolling_average_for_weight_date(day, 3), 2) if wr else None,
            "7-day avg": round(rolling_average_for_weight_date(day, 7), 2) if wr else None,
            "Breakfast": meal_text["Breakfast"],
            "Lunch": meal_text["Lunch"],
            "Dinner": meal_text["Dinner"],
            "Snacks": meal_text["Snacks"],
            "Calories": round(day_calories(day)),
            "Protein (g)": round(day_protein(day), 1),
            "Steps": get_steps(day),
            "Other activity": activity_names,
            "Activity kcal": round(day_activity_calories(day)) if day_activity_calories(day) else 0,
            "TDEE": round(tdee(day)) if tdee(day) is not None else None,
            "Daily deficit": round(daily_deficit(day)) if daily_deficit(day) is not None else None,
            "Cumulative deficit": round(cumulative.get(day, 0)),
        })

    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No tracking data yet.")


# =========================================================
# LOG DAY
# =========================================================

with tabs[1]:
    selected_day = st.date_input("Date", value=date.today(), key="log_day_date")
    selected_iso = selected_day.isoformat()

    left, right = st.columns(2)

    with left:
        st.subheader("Morning weigh-in")

        existing_weight = get_weight_record(selected_iso)
        default_weight = (
            float(existing_weight["weight"])
            if existing_weight
            else (current_weight() or 83.0)
        )

        with st.form("weight_form"):
            weight_value = st.number_input(
                "Weight (kg)",
                min_value=30.0,
                max_value=300.0,
                value=float(default_weight),
                step=0.05,
                format="%.2f",
            )
            submitted = st.form_submit_button("Save weight", use_container_width=True)

        if submitted:
            upsert_by_date("weights", {
                "date": selected_iso,
                "weight": round(float(weight_value), 2),
            })
            st.success("Weight saved.")
            st.rerun()

        st.subheader("Steps")

        existing_steps = get_steps(selected_iso)
        with st.form("steps_form"):
            steps_value = st.number_input(
                "Daily steps",
                min_value=0,
                max_value=100000,
                value=int(existing_steps or 0),
                step=100,
            )
            submitted_steps = st.form_submit_button(
                "Save steps",
                use_container_width=True,
            )

        if submitted_steps:
            upsert_by_date("steps", {
                "date": selected_iso,
                "steps": int(steps_value),
            })
            st.success("Steps saved.")
            st.rerun()

    with right:
        st.subheader("Other activity")

        activity_name = st.selectbox(
            "Activity",
            options=list(ACTIVITY_PRESETS.keys()),
            key="activity_name",
        )

        preset = ACTIVITY_PRESETS[activity_name]

        duration = None
        custom_calories = None

        if preset["type"] == "met":
            duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                max_value=600,
                value=30,
                step=5,
            )
        elif preset["type"] == "custom":
            custom_calories = st.number_input(
                "Additional activity calories",
                min_value=1.0,
                max_value=5000.0,
                value=200.0,
                step=10.0,
            )
        else:
            st.info(f'Fixed additional allowance: {preset["calories"]} kcal')

        if st.button("Add activity", use_container_width=True):
            if preset["type"] == "fixed":
                calories = float(preset["calories"])
                weight_used = None
            elif preset["type"] == "custom":
                calories = float(custom_calories)
                weight_used = None
            else:
                weight_used = get_weight_for_date(selected_iso)
                if weight_used is None:
                    st.error("Add at least one weigh-in before using MET activities.")
                    st.stop()
                calories = calculate_met_calories(
                    preset["met"],
                    weight_used,
                    duration,
                )

            data["activities"].append({
                "id": new_id(),
                "date": selected_iso,
                "name": activity_name,
                "calories": round(calories, 1),
                "duration": duration,
                "weight_used": weight_used,
            })
            save_data()
            st.success("Activity added.")
            st.rerun()

        day_acts = get_activities(selected_iso)
        if day_acts:
            st.markdown("**Activities logged**")
            for activity in day_acts:
                label = f'{activity["name"]} — {activity["calories"]:.0f} kcal'
                if activity.get("duration"):
                    label += f' ({activity["duration"]} min)'
                cols = st.columns([5, 1])
                cols[0].write(label)
                if cols[1].button("Delete", key=f'del_act_{activity["id"]}'):
                    delete_by_id("activities", activity["id"])
                    st.rerun()

    st.divider()
    st.subheader("Manual food entry")

    with st.form("manual_food_form", clear_on_submit=True):
        m1, m2 = st.columns(2)
        meal = m1.selectbox("Meal", MEALS, key="manual_meal")
        food_name = m2.text_input("Food", placeholder="Chicken breast")

        f1, f2, f3 = st.columns(3)
        quantity = f1.text_input("Quantity", placeholder="180 g")
        calories = f2.number_input("Calories", min_value=0.0, step=1.0)
        protein = f3.number_input("Protein (g)", min_value=0.0, step=0.1)

        manual_submit = st.form_submit_button(
            "Add food",
            use_container_width=True,
        )

    if manual_submit:
        if not food_name.strip():
            st.error("Enter a food name.")
        else:
            add_food_record(
                selected_iso,
                meal,
                food_name,
                calories,
                protein,
                quantity=quantity,
                source="manual",
            )
            st.success("Food added.")
            st.rerun()

    day_foods = get_foods(selected_iso)

    if day_foods:
        st.markdown("**Food logged for this day**")

        display = pd.DataFrame([
            {
                "Meal": f["meal"],
                "Food": f["name"],
                "Quantity": f.get("quantity", ""),
                "Calories": f["calories"],
                "Protein": f["protein"],
                "Source": f.get("source", ""),
                "ID": f["id"],
            }
            for f in day_foods
        ])

        st.dataframe(
            display.drop(columns=["ID"]),
            use_container_width=True,
            hide_index=True,
        )

        delete_food_id = st.selectbox(
            "Delete a food entry",
            options=[""] + [
                f'{f["id"]}|{f["meal"]}: {f["name"]}'
                for f in day_foods
            ],
            format_func=lambda x: "" if not x else x.split("|", 1)[1],
        )

        if delete_food_id and st.button("Delete selected food"):
            delete_by_id("foods", delete_food_id.split("|", 1)[0])
            st.rerun()

    st.divider()

    tdee_value = tdee(selected_iso)
    deficit_value = daily_deficit(selected_iso)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Calories", f"{day_calories(selected_iso):.0f} kcal")
    k2.metric("Protein", f"{day_protein(selected_iso):.1f} g")
    k3.metric("TDEE", f"{tdee_value:.0f} kcal" if tdee_value is not None else "—")
    k4.metric(
        "Deficit",
        f"{deficit_value:+.0f} kcal" if deficit_value is not None else "—",
    )

    with st.expander("Delete all data for this date"):
        st.warning("This removes the weigh-in, food, steps and activities for the selected date.")
        if st.button("Delete entire day", type="primary"):
            delete_day(selected_iso)
            st.rerun()


# =========================================================
# AI FOOD
# =========================================================

with tabs[2]:
    st.subheader("✨ Intelligent food entry")

    st.write(
        "Describe what you ate naturally. Gemini will use exact values you provide, "
        "your saved foods, and Google Search when useful, then return an editable estimate."
    )

    a1, a2 = st.columns([1, 1])
    ai_day = a1.date_input("Date", value=date.today(), key="ai_food_date")
    ai_meal = a2.selectbox("Meal", MEALS, key="ai_meal")

    food_description = st.text_area(
        "What did you eat?",
        height=120,
        placeholder=(
            "Example: 180g chicken breast, 200g hash browns at 159 kcal per 100g, "
            "broccoli and carrots"
        ),
    )

    if st.button("Analyse food", type="primary", use_container_width=True):
        if not food_description.strip():
            st.warning("Enter a food description first.")
        else:
            try:
                with st.spinner("Analysing food and checking nutrition values..."):
                    analysis, sources = analyse_food_with_gemini(food_description)

                st.session_state.ai_food_result = analysis.model_dump()
                st.session_state.ai_food_sources = sources
                st.session_state.ai_food_day = ai_day.isoformat()
                st.session_state.ai_food_meal = ai_meal
                st.success("Analysis complete.")
            except Exception as exc:
                st.error(f"Food analysis failed: {exc}")

    if "ai_food_result" in st.session_state:
        result = st.session_state.ai_food_result

        st.markdown("### Review before saving")
        st.caption(
            "Edit calories, protein, quantity or names below if anything looks wrong."
        )

        edit_df = pd.DataFrame(result["items"])

        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "calories": st.column_config.NumberColumn("Calories", min_value=0.0, step=1.0),
                "protein": st.column_config.NumberColumn("Protein (g)", min_value=0.0, step=0.1),
                "confidence": st.column_config.SelectboxColumn(
                    "Confidence",
                    options=["high", "medium", "low"],
                ),
                "source_type": st.column_config.SelectboxColumn(
                    "Source",
                    options=[
                        "user_provided",
                        "saved_food",
                        "web_researched",
                        "estimated",
                    ],
                ),
            },
            key="ai_editor",
        )

        total_cal = float(edited["calories"].fillna(0).sum()) if not edited.empty else 0
        total_pro = float(edited["protein"].fillna(0).sum()) if not edited.empty else 0

        m1, m2 = st.columns(2)
        m1.metric("Reviewed calories", f"{total_cal:.0f} kcal")
        m2.metric("Reviewed protein", f"{total_pro:.1f} g")

        if result.get("summary"):
            st.info(result["summary"])

        sources = st.session_state.get("ai_food_sources", [])
        if sources:
            with st.expander("Web sources used by Gemini"):
                for src in sources:
                    if src.get("uri"):
                        st.markdown(f'- [{src["title"]}]({src["uri"]})')
                    else:
                        st.write(src["title"])
        else:
            st.caption(
                "No Google Search source metadata was returned for this analysis; "
                "the result may have relied on supplied values, saved foods, or model estimation."
            )

        remember = st.checkbox(
            "Offer to save reviewed foods to my Saved Foods library after adding",
            value=False,
        )

        if st.button("Add reviewed items to diary", type="primary", use_container_width=True):
            target_day = st.session_state.get("ai_food_day", date.today().isoformat())
            target_meal = st.session_state.get("ai_food_meal", "Lunch")

            for _, row in edited.iterrows():
                if not str(row.get("name", "")).strip():
                    continue

                add_food_record(
                    target_day,
                    target_meal,
                    row.get("name", ""),
                    float(row.get("calories", 0) or 0),
                    float(row.get("protein", 0) or 0),
                    quantity=row.get("quantity", ""),
                    source=row.get("source_type", "estimated"),
                    confidence=row.get("confidence", ""),
                    assumption=row.get("assumption", ""),
                )

            if remember:
                st.session_state.remember_ai_foods = edited.to_dict("records")

            st.success(f"Added to {target_meal}.")
            del st.session_state.ai_food_result
            st.rerun()

    if "remember_ai_foods" in st.session_state:
        st.markdown("### Save analysed foods for quicker future matching")

        candidates = st.session_state.remember_ai_foods

        for i, item in enumerate(candidates):
            with st.expander(f'{item.get("name", "Food")} — {item.get("quantity", "")}'):
                st.write(
                    f'Current analysed portion: **{float(item.get("calories", 0)):.0f} kcal**, '
                    f'**{float(item.get("protein", 0)):.1f} g protein**'
                )

                if st.button("Save this exact portion", key=f"remember_{i}"):
                    data["saved_foods"].append({
                        "id": new_id(),
                        "name": item.get("name", ""),
                        "basis": "portion",
                        "quantity": item.get("quantity", ""),
                        "calories": round(float(item.get("calories", 0)), 1),
                        "protein": round(float(item.get("protein", 0)), 1),
                        "calories_per_100g": None,
                        "protein_per_100g": None,
                    })
                    save_data()
                    st.success("Saved.")

        if st.button("Finish saving foods"):
            del st.session_state.remember_ai_foods
            st.rerun()


# =========================================================
# SAVED FOODS
# =========================================================

with tabs[3]:
    st.subheader("⭐ Saved foods")

    st.write(
        "Save regular foods so Gemini can prefer your established nutrition values "
        "instead of estimating them again."
    )

    basis = st.radio(
        "Nutrition basis",
        ["Fixed portion", "Per 100 g"],
        horizontal=True,
    )

    with st.form("saved_food_form", clear_on_submit=True):
        s1, s2 = st.columns(2)
        saved_name = s1.text_input("Food name", placeholder="Yopro protein pudding")
        saved_quantity = s2.text_input(
            "Portion description",
            placeholder="130 g pot" if basis == "Fixed portion" else "100 g",
        )

        s3, s4 = st.columns(2)

        if basis == "Fixed portion":
            saved_cal = s3.number_input("Calories per portion", min_value=0.0, step=1.0)
            saved_pro = s4.number_input("Protein per portion (g)", min_value=0.0, step=0.1)
        else:
            saved_cal = s3.number_input("Calories per 100 g", min_value=0.0, step=1.0)
            saved_pro = s4.number_input("Protein per 100 g", min_value=0.0, step=0.1)

        saved_submit = st.form_submit_button(
            "Save food",
            use_container_width=True,
        )

    if saved_submit:
        if not saved_name.strip():
            st.error("Enter a food name.")
        else:
            if basis == "Fixed portion":
                record = {
                    "id": new_id(),
                    "name": saved_name.strip(),
                    "basis": "portion",
                    "quantity": saved_quantity.strip(),
                    "calories": round(saved_cal, 1),
                    "protein": round(saved_pro, 1),
                    "calories_per_100g": None,
                    "protein_per_100g": None,
                }
            else:
                record = {
                    "id": new_id(),
                    "name": saved_name.strip(),
                    "basis": "100g",
                    "quantity": "100 g",
                    "calories": None,
                    "protein": None,
                    "calories_per_100g": round(saved_cal, 1),
                    "protein_per_100g": round(saved_pro, 1),
                }

            data["saved_foods"].append(record)
            save_data()
            st.success("Saved food added.")
            st.rerun()

    if data["saved_foods"]:
        saved_rows = []
        for f in data["saved_foods"]:
            if f.get("basis") == "100g":
                nutrition = (
                    f'{f.get("calories_per_100g", 0):.0f} kcal / 100g, '
                    f'{f.get("protein_per_100g", 0):.1f}g protein / 100g'
                )
            else:
                nutrition = (
                    f'{f.get("calories", 0):.0f} kcal, '
                    f'{f.get("protein", 0):.1f}g protein'
                )

            saved_rows.append({
                "Food": f["name"],
                "Basis": "Per 100g" if f.get("basis") == "100g" else "Fixed portion",
                "Quantity": f.get("quantity", ""),
                "Nutrition": nutrition,
                "ID": f["id"],
            })

        st.dataframe(
            pd.DataFrame(saved_rows).drop(columns=["ID"]),
            use_container_width=True,
            hide_index=True,
        )

        selected_saved = st.selectbox(
            "Delete saved food",
            options=[""] + [
                f'{f["id"]}|{f["name"]}'
                for f in data["saved_foods"]
            ],
            format_func=lambda x: "" if not x else x.split("|", 1)[1],
        )

        if selected_saved and st.button("Delete selected saved food"):
            delete_by_id("saved_foods", selected_saved.split("|", 1)[0])
            st.rerun()
    else:
        st.info("No saved foods yet.")


# =========================================================
# DATA / BACKUP
# =========================================================

with tabs[4]:
    st.subheader("💾 Backup and restore")

    st.warning(
        "This Version 5 build stores data in a local JSON file. "
        "That is convenient for testing, but hosted Streamlit storage should not be treated "
        "as your long-term database. Use the export regularly until we connect permanent storage."
    )

    export_json = json.dumps(data, indent=2, ensure_ascii=False)

    st.download_button(
        "Download tracker backup",
        data=export_json,
        file_name=f"weight_tracker_backup_{date.today().isoformat()}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = st.file_uploader(
        "Restore from a tracker JSON backup",
        type=["json"],
    )

    if uploaded is not None:
        if st.button("Restore uploaded backup", type="primary"):
            try:
                restored = json.load(uploaded)
                required = {"weights", "foods", "steps", "activities", "saved_foods", "settings"}
                if not required.issubset(restored.keys()):
                    raise ValueError("Backup is missing required tracker sections.")

                st.session_state.data = restored
                DATA_FILE.write_text(
                    json.dumps(restored, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                st.success("Backup restored.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not restore backup: {exc}")

    st.divider()
    st.subheader("Storage summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Weigh-ins", len(data["weights"]))
    s2.metric("Food entries", len(data["foods"]))
    s3.metric("Activity entries", len(data["activities"]))
    s4.metric("Saved foods", len(data["saved_foods"]))
