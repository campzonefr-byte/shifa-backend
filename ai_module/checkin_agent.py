from datetime import date
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from food_db import estimate_meal_calories

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def extract_checkin_with_llm(meals_today: str, activity_today: str, mood: str = "") -> dict:
    prompt = f"""
You are a wellness data extraction assistant.

Extract structured signals from the user's daily check-in.

Return ONLY valid JSON with this exact schema:
{{
  "food_patterns": [],
  "food_items": [],
  "activity_level_today": "low/moderate/high/unknown",
  "activity_duration_minutes": null,
  "health_interests": [],
  "last_detected_issue": null,
  "meal_summary": "",
  "activity_summary": "",
  "mood": ""
}}

food_items: list of foods/dishes mentioned, one entry per dish, as:
  {{"name": "<dish name, keep original language>", "portion": "small/medium/large"}}
  - Infer portion from context (e.g. "beaucoup" -> large, "un peu" -> small).
  - If portion is not mentioned at all, use "medium".
  - Split combined meals into separate items (e.g. "couscous poulet ou salade" -> two items).

Allowed food_patterns:
- heavy_meals
- high_sugar
- vegetable_intake
- protein_intake
- light_meals
- balanced_meals

Allowed health_interests:
- digestion
- bloating_gas
- constipation
- fiber_support
- appetite_control
- detox
- respiratory_support
- smoking_reduction_support
- glycemic_balance
- insulin_sensitivity
- sugar_cravings
- weight_loss
- muscle_gain
- blood_regulation
- heart_support
- stress_anxiety
- sleep

Rules:
- If user mentions gas, bloating, constipation, stomach discomfort, digestion issues → health_interests includes digestion
- If user mentions liver, detox, toxins → detox
- If user mentions low activity/no activity → activity_level_today = low
- If walking/light movement → moderate
- If gym/running/intense sport → high
- Extract activity duration in minutes if mentioned. If not mentioned, return null.

User meals today:
{meals_today}

User activity today:
{activity_today}

Mood:
{mood}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON. No explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception:
        return {
            "food_patterns": detect_food_patterns(meals_today),
            "food_items": [],
            "activity_level_today": detect_activity_level(activity_today),
            "activity_duration_minutes": None,
            "health_interests": detect_health_interests(meals_today),
            "last_detected_issue": None,
            "meal_summary": meals_today,
            "activity_summary": activity_today,
            "mood": mood,
        }

def estimate_daily_energy_with_llm(
    user_profile: dict,
    meals_today: str,
    activity_today: str,
    mood: str = "",
    food_items: list | None = None,
) -> dict:
    language = user_profile.get("language", "ar")

    # --- Step 1: local nutrition table for known dishes (deterministic) ---
    local_result = estimate_meal_calories(food_items or [])
    known_calories = local_result["known_calories_total"]
    unresolved_items = local_result["items_needing_llm_estimate"]
    table_confidence = local_result["confidence"]

    # --- Step 2: LLM only fills the gaps (unknown foods + activity burn +
    # a short human insight). Much smaller/cheaper prompt than before, and
    # the calorie figure for known dishes no longer varies between calls. ---
    prompt = f"""
You are a wellness estimation assistant.

Some foods in this check-in were already matched against a local nutrition
table and their calories are already known (see "already_known_calories"
below) — do NOT re-estimate those, only handle the items listed under
"unresolved_food_items".

Return ONLY valid JSON.

Language for text fields: {language}

User profile:
{json.dumps(user_profile, ensure_ascii=False)}

Already known calories (from local table, do not change): {known_calories}
Unresolved food items needing an estimate: {unresolved_items}

Activity today:
{activity_today}

Mood:
{mood}

Return this JSON schema:
{{
  "unresolved_items_calories": null,
  "estimated_calories_burned": null,
  "assumptions": [],
  "insight_message": ""
}}

Rules:
- unresolved_items_calories: sum of realistic calorie estimates for ONLY
  the items in "unresolved_food_items" (0 or null if that list is empty).
- Use realistic approximate values for activity calories burned.
- Do not present estimates as exact medical facts.
- If data is missing, use null and explain in assumptions.
- If language is fr, write text fields in French.
- If language is ar, write text fields in Arabic/Tunisian Arabic.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        llm_part = json.loads(response.choices[0].message.content.strip())

    except Exception:
        llm_part = {
            "unresolved_items_calories": None,
            "estimated_calories_burned": None,
            "assumptions": ["Energy estimation unavailable for unresolved items."],
            "insight_message": ""
        }

    unresolved_calories = llm_part.get("unresolved_items_calories") or 0
    estimated_calories_in = (known_calories + unresolved_calories) or None
    estimated_calories_burned = llm_part.get("estimated_calories_burned")

    estimated_net_calories = None
    estimated_weekly_weight_change_kg = None
    weight_trend = None
    prediction_available = False

    if estimated_calories_in is not None and estimated_calories_burned is not None:
        estimated_net_calories = estimated_calories_in - estimated_calories_burned
        estimated_weekly_weight_change_kg = round((estimated_net_calories * 7) / 7700, 2)
        if estimated_net_calories > 200:
            weight_trend = "possible gain"
        elif estimated_net_calories < -200:
            weight_trend = "possible loss"
        else:
            weight_trend = "stable"

    return {
        "estimated_calories_in": estimated_calories_in,
        "estimated_calories_burned": estimated_calories_burned,
        "estimated_net_calories": estimated_net_calories,
        "estimated_weekly_weight_change_kg": estimated_weekly_weight_change_kg,
        "weight_trend": weight_trend,
        "assumptions": llm_part.get("assumptions", []),
        "insight_message": llm_part.get("insight_message", ""),
        "calorie_estimate_confidence": table_confidence,
        "calorie_breakdown": local_result["items"],
    }

def detect_food_patterns(meals_text: str) -> list:
    text = (meals_text or "").lower()
    patterns = []

    if any(w in text for w in [
        "pizza", "burger", "fried", "frite", "fast food", "couscous", "pate", "maqarouna", "mokli", "khobz", "mloukheya", "rfissa", "fast food",
        "makla th9ila", "مقلي", "ماكلة ثقيلة"
    ]):
        patterns.append("heavy_meals")

    if any(w in text for w in [
        "soda", "jus", "juice", "cake", "chocolate", "gateau", "makroudh", "ghrayba", "crepe",
        "sweet", "sucre", "حلويات", "سكر"
    ]):
        patterns.append("high_sugar")

    if any(w in text for w in [
        "salad", "vegetables", "khodhra", "خضرة", "jus frais", "carotte", "laitu",
        "soupe", "soup", "سلطة", "شربة"
    ]):
        patterns.append("vegetable_intake")

    if any(w in text for w in [
        "chicken", "egg", "fish", "tuna", "meat", "protein",
        "دجاج", "بيض", "حوت", "لحم", "تونة"
    ]):
        patterns.append("protein_intake")

    return patterns


def detect_health_interests(meals_text: str) -> list:
    text = (meals_text or "").lower()
    interests = []

    if any(w in text for w in [
        "gaz", "gas", "bloating", "constipation", "kerch",
        "نفخة", "غازات", "إمساك", "كرش", "هضم"
    ]):
        interests.append("digestion")

    if any(w in text for w in [
        "detox", "liver", "kebda", "كبد", "سموم"
    ]):
        interests.append("detox")

    return interests


def detect_activity_level(activity_text: str) -> str:
    text = (activity_text or "").lower()

    if any(w in text for w in [
        "nothing", "no activity", "rest", "sedentary",
        "ma 3malt chay", "ما عملت شي", "ما عملتش", "رقدت"
    ]):
        return "low"

    if any(w in text for w in [
        "walk", "walking", "marche", "steps", "light",
        "20 min", "30 min", "مشي", "مشيت", "dance"
    ]):
        return "moderate"

    if any(w in text for w in [
        "gym", "training", "running", "cycling", "swimming",
        "sport", "intense", "جيم", "رياضة", "سباحة", "جريت"
    ]):
        return "high"

    return "unknown"



def calculate_bmr(weight, height, age, sex):
    if not weight or not height or not age or not sex:
        return None

    sex = sex.lower()

    if sex == "male":
        return round(10 * weight + 6.25 * height - 5 * age + 5)

    if sex == "female":
        return round(10 * weight + 6.25 * height - 5 * age - 161)

    return None

def get_activity_factor(activity_level: str):
    factors = {
        "low": 1.2,
        "moderate": 1.375,
        "high": 1.55,
        "unknown": 1.2,
    }
    return factors.get(activity_level, 1.2)


def calculate_tdee(bmr, activity_level):
    if bmr is None:
        return None

    factor = get_activity_factor(activity_level)
    return round(bmr * factor)

def estimate_activity_calories(weight, duration_minutes, intensity):
    if not weight or not duration_minutes:
        return None

    met_values = {
        "low": 2.5,
        "moderate": 4.0,
        "high": 7.0,
        "unknown": 3.0,
    }

    met = met_values.get(intensity, 3.0)

    calories = met * weight * (duration_minutes / 60)
    return round(calories)
def estimate_steps_calories(weight, steps):
    if not weight or not steps:
        return 0

    try:
        steps = int(steps)
        weight = float(weight)
    except (TypeError, ValueError):
        return 0

    if steps <= 0:
        return 0

    # Approximate calorie burn based on body weight
    calories_per_step = 0.0005 * weight

    return round(steps * calories_per_step)

def structured_energy_calculation(
    user_profile: dict,
    extracted: dict,
    steps: int = 0,
) -> dict:
    weight = user_profile.get("weight")
    height = user_profile.get("height")
    age = user_profile.get("age")
    sex = user_profile.get("sex")

    activity_level = extracted.get(
        "activity_level_today",
        "unknown",
    )

    duration = extracted.get(
        "activity_duration_minutes"
    )

    bmr = calculate_bmr(
        weight,
        height,
        age,
        sex,
    )

    sedentary_tdee = (
        round(bmr * 1.2)
        if bmr is not None
        else None
    )

    workout_calories = (
        estimate_activity_calories(
            weight=weight,
            duration_minutes=duration,
            intensity=activity_level,
        )
        or 0
    )

    steps_calories = estimate_steps_calories(
        weight=weight,
        steps=steps,
    )

    total_activity_calories = (
        workout_calories + steps_calories
    )

    total_calories_burned = (
        sedentary_tdee + total_activity_calories
        if sedentary_tdee is not None
        else None
    )

    return {
        "bmr": bmr,
        "sedentary_tdee": sedentary_tdee,
        "tdee": total_calories_burned,

        "estimated_workout_calories_burned": workout_calories,
        "estimated_steps_calories_burned": steps_calories,
        "estimated_activity_calories_burned": total_activity_calories,

        "steps": steps,
        "activity_duration_minutes": duration,
        "activity_intensity": activity_level,
    }

def compute_consistency_from_recent_checkins(recent_checkins: list) -> dict:
    if not recent_checkins:
        return {
            "score": 50,
            "level": "medium",
            "summary": "Not enough check-in history yet."
        }

    score = 50

    checkin_count = len(recent_checkins)

    # More check-ins = better consistency
    score += min(checkin_count * 5, 20)

    low_activity_days = 0
    active_days = 0
    high_calorie_days = 0

    for checkin in recent_checkins:
        activities = checkin.get("daily_activity_logs", []) or []
        meals = checkin.get("daily_meal_logs", []) or []

        if activities:
            for act in activities:
                intensity = act.get("intensity")
                if intensity in ["moderate", "high"]:
                    active_days += 1
                elif intensity == "low":
                    low_activity_days += 1
        else:
            low_activity_days += 1

        for meal in meals:
            calories = meal.get("estimated_calories")
            if calories and calories >= 2200:
                high_calorie_days += 1

    score += active_days * 5
    score -= low_activity_days * 4
    score -= high_calorie_days * 5

    score = max(0, min(score, 100))

    if score < 40:
        level = "low"
    elif score < 70:
        level = "medium"
    else:
        level = "high"

    return {
        "score": score,
        "level": level,
        "summary": f"Based on the last {checkin_count} check-ins."
    }

def analyze_recent_trends(recent_checkins: list) -> dict:
    if not recent_checkins:
        return {
            "sugar_frequency": 0,
            "heavy_meal_frequency": 0,
            "low_activity_days": 0,
            "active_days": 0,
            "sedentary_streak": 0,
            "trend_direction": "unknown",
            "insight": "Not enough history yet."
        }

    sugar_frequency = 0
    heavy_meal_frequency = 0
    low_activity_days = 0
    active_days = 0
    sedentary_streak = 0
    current_sedentary_streak = 0

    total_calories = []

    for checkin in recent_checkins:
        meals = checkin.get("daily_meal_logs", []) or []
        activities = checkin.get("daily_activity_logs", []) or []

        day_low_activity = True

        for meal in meals:
            description = (meal.get("description") or "").lower()
            calories = meal.get("estimated_calories")

            if calories:
                total_calories.append(calories)

            if any(w in description for w in [
                "soda", "jus", "juice", "cake", "chocolate", "sweet",
                "sucre", "gâteau", "gateau", "coca", "حلويات", "سكر"
            ]):
                sugar_frequency += 1

            if any(w in description for w in [
                "pizza", "burger", "fried", "frite", "fast food",
                "couscous", "pate", "maqarouna", "مقلي", "ماكلة ثقيلة"
            ]):
                heavy_meal_frequency += 1

        for act in activities:
            intensity = act.get("intensity")

            if intensity in ["moderate", "high"]:
                active_days += 1
                day_low_activity = False

        if day_low_activity:
            low_activity_days += 1
            current_sedentary_streak += 1
        else:
            sedentary_streak = max(sedentary_streak, current_sedentary_streak)
            current_sedentary_streak = 0

    sedentary_streak = max(sedentary_streak, current_sedentary_streak)

    avg_calories = round(sum(total_calories) / len(total_calories), 1) if total_calories else None

    if sugar_frequency >= 4 or heavy_meal_frequency >= 4 or sedentary_streak >= 3:
        trend_direction = "worsening"
    elif active_days >= 4 and sugar_frequency <= 2 and heavy_meal_frequency <= 2:
        trend_direction = "improving"
    else:
        trend_direction = "stable"

    insight = (
        f"Based on the last {len(recent_checkins)} check-ins: "
        f"sugar frequency={sugar_frequency}, heavy meals={heavy_meal_frequency}, "
        f"low activity days={low_activity_days}, sedentary streak={sedentary_streak}."
    )

    return {
        "sugar_frequency": sugar_frequency,
        "heavy_meal_frequency": heavy_meal_frequency,
        "low_activity_days": low_activity_days,
        "active_days": active_days,
        "sedentary_streak": sedentary_streak,
        "average_calories": avg_calories,
        "trend_direction": trend_direction,
        "insight": insight
    }

def build_daily_checkin_output(
    user_profile: dict,
    meals_today: str,
    activity_today: str,
    mood: str = "",
    steps: int = 0,
) -> dict:
    user_profile = user_profile or {}
    goals = user_profile.get("goals", [])
    goals_text = " ".join(goals).lower() if isinstance(goals, list) else str(goals).lower()

    extracted = extract_checkin_with_llm(meals_today, activity_today, mood)
    structured_energy = structured_energy_calculation(
        user_profile=user_profile,
        extracted=extracted,
        steps=steps,
    )
    energy_estimation = estimate_daily_energy_with_llm(
        user_profile,
        meals_today,
        activity_today,
        mood,
        food_items=extracted.get("food_items", []),
    )
    
    bmr = structured_energy.get("bmr")
    tdee = structured_energy.get("tdee")
    activity_burned = structured_energy.get("estimated_activity_calories_burned") or 0

    if tdee is not None:
        energy_estimation["estimated_calories_burned"] = tdee
        energy_estimation["estimated_total_calories_burned"] = tdee
        energy_estimation["estimated_bmr"] = bmr
        energy_estimation["estimated_tdee"] = tdee
        energy_estimation["estimated_activity_calories_burned"] = activity_burned

    if (
        energy_estimation.get("estimated_calories_in") is not None
        and energy_estimation.get("estimated_calories_burned") is not None
    ):

        net = (
            energy_estimation["estimated_calories_in"]
            - energy_estimation["estimated_calories_burned"]
        )

        energy_estimation["estimated_net_calories"] = net

    # Don't predict weekly weight change if today's food log is probably incomplete
        if energy_estimation["estimated_calories_in"] < 800:

            energy_estimation["estimated_weekly_weight_change_kg"] = None
            energy_estimation["weight_trend"] = "unknown"
            energy_estimation["prediction_available"] = False

            assumptions = energy_estimation.get("assumptions", [])
            language = user_profile.get("language", "ar")

            if language == "fr":
                message = (
                    "L'apport alimentaire de la journée semble incomplet. "
                    "La prédiction hebdomadaire du poids n'a pas été calculée."
                )

            elif language == "en":
                message = (
                    "Today's food intake appears incomplete. "
                    "Weekly weight prediction was not calculated."
                )

            else:
                message = (
                    "يبدو أن تسجيل الوجبات لليوم غير مكتمل، لذلك لم يتم حساب توقع تغير الوزن الأسبوعي."
                )

            assumptions.append(message)
            energy_estimation["assumptions"] = assumptions

        else:
            energy_estimation["prediction_available"] = True
            weekly_change = round((net * 7) / 7700, 2)

            energy_estimation["estimated_weekly_weight_change_kg"] = weekly_change

            if net > 200:
                energy_estimation["weight_trend"] = "possible gain"
            elif net < -200:
                energy_estimation["weight_trend"] = "possible loss"
            else:
                energy_estimation["weight_trend"] = "stable"

    food_patterns = extracted.get("food_patterns", [])
    health_interests = extracted.get("health_interests", [])
    activity_level_today = extracted.get("activity_level_today", "unknown")
    last_detected_issue = extracted.get("last_detected_issue")
    meal_summary = extracted.get("meal_summary", meals_today)
    activity_summary = extracted.get("activity_summary", activity_today)

    product_hint = None

    

    if "digestion" in health_interests:
        product_hint = "Colon Detox"
    elif "detox" in health_interests:
        product_hint = "Liver Detox"
    elif ("weight loss" in goals_text or "weightloss" in goals_text) and activity_level_today == "low":
        product_hint = "Slim Pack"

    consistency_score = 0

    if meals_today.strip():
        consistency_score += 40

    if activity_level_today == "high":
        consistency_score += 40
    elif activity_level_today == "moderate":
        consistency_score += 30
    elif activity_level_today == "low":
        consistency_score += 15

    if mood:
        consistency_score += 20

    notes = [f"Daily check-in on {date.today()}"]
    if mood:
        notes.append(f"Mood: {mood}")

    memory_updates = {
        "health_interests": health_interests,
        "recurring_food_patterns": food_patterns,
        "recurring_activity_patterns": [activity_level_today] if activity_level_today != "unknown" else [],
        "last_meal_summary": meal_summary,
        "last_activity_summary": activity_summary,
        "last_detected_issue": last_detected_issue or (health_interests[0] if health_interests else None),
        "consistency_score": consistency_score,
        "notes": notes,
    }

    if product_hint:
        memory_updates["last_recommended_product"] = product_hint
        memory_updates["past_recommended_products"] = [product_hint]

    return {
        "checkin_date": str(date.today()),
        "food_patterns": food_patterns,
        "health_interests": health_interests,
        "activity_level_today": activity_level_today,
      
        "product_hint": product_hint,
        "consistency_score": consistency_score,
        "energy_estimation": energy_estimation,
        "structured_energy": structured_energy,
        "memory_updates": memory_updates,
        
    }