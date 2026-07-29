from ai_module.supabase_client import supabase
from datetime import datetime, timedelta
from datetime import date, timedelta
from ai_module.supabase_client import supabase

def get_or_create_user(user_code: str):
    user_code = str(
        user_code or "demo_user"
    ).strip()

    # Remove accidental surrounding quotation marks
    user_code = user_code.strip('"').strip("'")

    if not user_code:
        user_code = "demo_user"

    existing = (
        supabase.table("users")
        .select("*")
        .eq("user_code", user_code)
        .execute()
    )

    if existing.data:
        return existing.data[0]

    created = (
        supabase.table("users")
        .insert({
            "user_code": user_code
        })
        .execute()
    )

    return created.data[0]


def get_user_memory(user_code: str):
    user = get_or_create_user(user_code)

    existing = (
        supabase.table("user_memory")
        .select("*")
        .eq("user_id", user["id"])
        .execute()
    )

    if existing.data:
        return existing.data[0]

    created = (
        supabase.table("user_memory")
        .insert({
            "user_id": user["id"],
            "health_interests": [],
            "past_recommended_products": [],
            "recurring_food_patterns": [],
            "recurring_activity_patterns": [],
            "notes": [],
        })
        .execute()
    )

    return created.data[0]

def get_recent_chat_history(user_code: str, days: int = 90, limit: int = 100):
    from datetime import datetime, timedelta

    user = get_or_create_user(user_code)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    res = (
        supabase.table("chat_interactions")
        .select("*")
        .eq("user_id", user["id"])
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return res.data or []

def merge_unique(existing, new_items):
    existing = existing or []

    if not isinstance(new_items, list):
        new_items = [new_items]

    result = list(existing)

    for item in new_items:
        if item and item not in result:
            result.append(item)

    return result

def log_chat_interaction(user_code: str, data: dict):
    user = get_or_create_user(user_code)

    row = {
        "user_id": user["id"],
        "question": data.get("question"),
        "answer": data.get("answer"),
        "intent": data.get("intent"),
        "detected_product": data.get("detected_product"),
        "recommended_product": data.get("recommended_product"),
        "used_memory": data.get("used_memory", False),
    }

    supabase.table("chat_interactions").insert(row).execute()

def log_recommendation(user_code: str, data: dict):
    user = get_or_create_user(user_code)

    row = {
        "user_id": user["id"],
        "recommended_products": data.get("recommended_products", []),
        "meal_recommendations": data.get("meal_recommendations", []),
        "exercise_recommendations": data.get("exercise_recommendations", []),
        "reasoning_summary": data.get("reasoning_summary"),

        "used_memory": data.get("used_memory", False),
        "used_goal": data.get("used_goal", False),
        "used_activity": data.get("used_activity", False),
        "used_bmi": data.get("used_bmi", False),
        "used_daily_checkin": data.get("used_daily_checkin", False),

        "response_time_sec": data.get("response_time_sec"),
    }

    supabase.table("recommendation_logs").insert(row).execute()


def create_daily_checkin(
    user_code: str,
    data: dict,
) -> dict:
    user = get_or_create_user(user_code)

    checkin_date = (
        data.get("checkin_date")
        or str(date.today())
    )

    existing = (
        supabase.table("daily_checkins")
        .select("*")
        .eq("user_id", user["id"])
        .eq("checkin_date", checkin_date)
        .execute()
    )

    base_row = {
        "user_id": user["id"],
        "checkin_date": checkin_date,
        "mood": data.get("mood"),
        "energy_level": data.get(
            "energy_level"
        ),
        "daily_steps": data.get(
            "daily_steps",
            0,
        ),
        "consistency_score": data.get(
            "consistency_score",
            0,
        ),
    }

    if existing.data:
        current = existing.data[0]

        update_row = {
            "mood": (
                data.get("mood")
                or current.get("mood")
            ),
            "energy_level": (
                data.get("energy_level")
                or current.get("energy_level")
            ),
            "daily_steps": max(
                int(
                    data.get("daily_steps")
                    or 0
                ),
                int(
                    current.get("daily_steps")
                    or 0
                ),
            ),
            "consistency_score": data.get(
                "consistency_score",
                current.get(
                    "consistency_score",
                    0,
                ),
            ),
        }

        result = (
            supabase.table("daily_checkins")
            .update(update_row)
            .eq("id", current["id"])
            .execute()
        )

        return result.data[0]

    result = (
        supabase.table("daily_checkins")
        .insert(base_row)
        .execute()
    )

    return result.data[0]

    return result.data[0]
def log_daily_meal(
    checkin_id: int,
    data: dict,
) -> dict | None:
    description = data.get("description")

    if not description:
        return None

    row = {
        "checkin_id": checkin_id,
        "meal_type": data.get(
            "meal_type",
            "general",
        ),
        "description": description,
        "estimated_calories": data.get(
            "estimated_calories"
        ),
        "estimated_protein": data.get(
            "estimated_protein"
        ),
    }

    result = (
        supabase.table("daily_meal_logs")
        .insert(row)
        .execute()
    )

    return result.data[0] if result.data else None


def log_daily_activity(
    checkin_id: int,
    data: dict,
) -> dict | None:
    description = (
        data.get("description")
        or data.get("activity_type")
    )

    duration = data.get("duration_minutes")

    if not description and not duration:
        return None

    row = {
        "checkin_id": checkin_id,
        "activity_type": (
            description or "general"
        ),
        "duration_minutes": duration,
        "intensity": data.get("intensity"),
        "estimated_calories_burned": data.get(
            "estimated_calories_burned"
        ),
    }

    result = (
        supabase.table("daily_activity_logs")
        .insert(row)
        .execute()
    )

    return result.data[0] if result.data else None

def get_daily_checkin(
    user_code: str,
    checkin_date: str | None = None,
) -> dict | None:
    user = get_or_create_user(user_code)

    target_date = (
        checkin_date
        or str(date.today())
    )

    result = (
        supabase.table("daily_checkins")
        .select("*")
        .eq("user_id", user["id"])
        .eq("checkin_date", target_date)
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    checkin = result.data[0]

    meals_result = (
        supabase.table("daily_meal_logs")
        .select("*")
        .eq("checkin_id", checkin["id"])
        .execute()
    )

    activities_result = (
        supabase.table("daily_activity_logs")
        .select("*")
        .eq("checkin_id", checkin["id"])
        .execute()
    )

    checkin["daily_meal_logs"] = (
        meals_result.data or []
    )

    checkin["daily_activity_logs"] = (
        activities_result.data or []
    )

    return checkin

def calculate_bmr(
    weight: float | None,
    height: float | None,
    age: int | None,
    sex: str | None,
) -> int | None:
    if not all([
        weight,
        height,
        age,
        sex,
    ]):
        return None

    normalized_sex = str(sex).lower()

    if normalized_sex in {
        "male",
        "man",
        "homme",
        "m",
    }:
        return round(
            10 * weight
            + 6.25 * height
            - 5 * age
            + 5
        )

    if normalized_sex in {
        "female",
        "woman",
        "femme",
        "f",
    }:
        return round(
            10 * weight
            + 6.25 * height
            - 5 * age
            - 161
        )

    return None

def calculate_daily_summary(
    checkin: dict,
    user_profile: dict,
) -> dict:
    meals = (
        checkin.get("daily_meal_logs")
        or []
    )

    activities = (
        checkin.get("daily_activity_logs")
        or []
    )

    calories_in = round(
        sum(
            float(
                meal.get("estimated_calories")
                or 0
            )
            for meal in meals
        ),
        1,
    )

    activity_calories = round(
        sum(
            float(
                activity.get(
                    "estimated_calories_burned"
                )
                or 0
            )
            for activity in activities
        ),
        1,
    )

    activity_minutes = sum(
        int(
            activity.get("duration_minutes")
            or 0
        )
        for activity in activities
    )

    bmr = calculate_bmr(
        user_profile.get("weight"),
        user_profile.get("height"),
        user_profile.get("age"),
        user_profile.get("sex"),
    )

    sedentary_tdee = (
        round(bmr * 1.2)
        if bmr is not None
        else None
    )

    total_calories_burned = None
    net_calories = None
    weekly_change = None
    weight_trend = "unknown"

    if sedentary_tdee is not None:
        total_calories_burned = round(
            sedentary_tdee
            + activity_calories,
            1,
        )

        net_calories = round(
            calories_in
            - total_calories_burned,
            1,
        )

        weekly_change = round(
            (net_calories * 7) / 7700,
            2,
        )

        if net_calories > 200:
            weight_trend = "possible gain"
        elif net_calories < -200:
            weight_trend = "possible loss"
        else:
            weight_trend = "stable"

    return {
        "checkin_id": checkin["id"],
        "checkin_date": checkin["checkin_date"],
        "checkin_count": (
            len(meals) + len(activities)
        ),
        "meal_count": len(meals),
        "activity_count": len(activities),
        "calories_in": calories_in,
        "activity_calories_burned": (
            activity_calories
        ),
        "activity_minutes": activity_minutes,
        "bmr": bmr,
        "sedentary_tdee": sedentary_tdee,
        "total_calories_burned": (
            total_calories_burned
        ),
        "net_calories": net_calories,
        "estimated_weekly_weight_change_kg": (
            weekly_change
        ),
        "weight_trend": weight_trend,
    }

def update_daily_summary(
    user_code: str,
    user_profile: dict,
    checkin_date: str | None = None,
) -> dict | None:
    checkin = get_daily_checkin(
        user_code,
        checkin_date,
    )

    if not checkin:
        return None

    summary = calculate_daily_summary(
        checkin,
        user_profile,
    )

    update_row = {
        "calories_in": summary["calories_in"],
        "activity_calories_burned": (
            summary[
                "activity_calories_burned"
            ]
        ),
        "sedentary_tdee": summary[
            "sedentary_tdee"
        ],
        "total_calories_burned": summary[
            "total_calories_burned"
        ],
        "net_calories": summary[
            "net_calories"
        ],
        "estimated_weekly_weight_change_kg": (
            summary[
                "estimated_weekly_weight_change_kg"
            ]
        ),
        "weight_trend": summary[
            "weight_trend"
        ],
        "activity_minutes": summary[
            "activity_minutes"
        ],
        "checkin_count": summary[
            "checkin_count"
        ],
    }

    (
        supabase.table("daily_checkins")
        .update(update_row)
        .eq("id", checkin["id"])
        .execute()
    )

    return summary

def get_checkins_between(
    user_code: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    user = get_or_create_user(user_code)

    result = (
        supabase.table("daily_checkins")
        .select("*")
        .eq("user_id", user["id"])
        .gte("checkin_date", start_date)
        .lte("checkin_date", end_date)
        .order("checkin_date")
        .execute()
    )

    return result.data or []

def get_last_days_checkins(
    user_code: str,
    days: int = 7,
) -> list[dict]:
    end = date.today()
    start = end - timedelta(
        days=days - 1
    )

    return get_checkins_between(
        user_code,
        str(start),
        str(end),
    )
def log_weight(
    user_code: str,
    weight: float,
) -> dict:
    if weight <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    user = get_or_create_user(user_code)

    result = (
        supabase.table("weight_logs")
        .insert({
            "user_id": user["id"],
            "weight": weight,
        })
        .execute()
    )

    profile = get_user_profile(user_code)

    if profile:
        (
            supabase.table("user_profiles")
            .update({
                "weight": weight,
            })
            .eq("id", profile["id"])
            .execute()
        )

    return result.data[0]

def get_weight_history(
    user_code: str,
    limit: int = 90,
) -> list[dict]:
    user = get_or_create_user(user_code)

    result = (
        supabase.table("weight_logs")
        .select("weight, logged_at")
        .eq("user_id", user["id"])
        .order("logged_at")
        .limit(limit)
        .execute()
    )

    return result.data or []

def get_loyalty(user_code: str) -> dict:
    user = get_or_create_user(user_code)

    result = (
        supabase
        .table("loyalty")
        .select("*")
        .eq("user_id", user["id"])
        .execute()
    )

    if result.data:
        return result.data[0]

    new_loyalty = {
        "user_id": user["id"],
        "points": 0,
        "level": "Bronze",
    }

    created = (
        supabase
        .table("loyalty")
        .insert(new_loyalty)
        .execute()
    )

    if created.data:
        return created.data[0]

    return new_loyalty

def add_loyalty_points(
    user_code: str,
    points: int,
) -> dict:
    loyalty = get_loyalty(user_code)

    new_points = (
        int(loyalty.get("points") or 0)
        + int(points)
    )

    result = (
        supabase.table("loyalty")
        .update({
            "points": new_points,
        })
        .eq("id", loyalty["id"])
        .execute()
    )

    return result.data[0]

def get_recent_checkins(user_code: str, limit: int = 50):
    user = get_or_create_user(user_code)

    since = (datetime.utcnow() - timedelta(days=7)).isoformat()

    checkins_res = (
        supabase.table("daily_checkins")
        .select("*")
        .eq("user_id", user["id"])
        .gte("created_at", since)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    checkins = checkins_res.data or []

    for checkin in checkins:
        meals_res = (
            supabase.table("daily_meal_logs")
            .select("*")
            .eq("checkin_id", checkin["id"])
            .execute()
        )

        activities_res = (
            supabase.table("daily_activity_logs")
            .select("*")
            .eq("checkin_id", checkin["id"])
            .execute()
        )

        checkin["daily_meal_logs"] = meals_res.data or []
        checkin["daily_activity_logs"] = activities_res.data or []

    return checkins

def get_user_profile(user_code: str) -> dict:
    user = get_or_create_user(user_code)

    profile_id = user.get("profile_id")

    if not profile_id:
        return {}

    res = (
        supabase.table("user_profiles")
        .select("*")
        .eq("id", profile_id)
        .execute()
    )

    if res.data:
        return res.data[0]

    return {}

def upsert_user_profile(
    user_code: str,
    profile_data: dict,
):
    user = get_or_create_user(user_code)

    row = {
        "name": (
            profile_data.get("name")
            or user_code
        ),
        "age": profile_data.get("age"),
        "height": profile_data.get("height"),
        "height_unit": "cm",
        "weight": profile_data.get("weight"),
        "starting_weight": profile_data.get(
            "starting_weight"
        ),
        "target_weight": profile_data.get(
            "target_weight"
        ),
        "weight_unit": "kg",
        "sex": profile_data.get("sex"),
        "goals": profile_data.get(
            "goals",
            [],
        ),
        "language": profile_data.get(
            "language",
            "ar",
        ),
        "medical_conditions": (
            profile_data.get(
                "medical_conditions",
                [],
            )
        ),
    }

    if user.get("profile_id"):
        res = (
            supabase
            .table("user_profiles")
            .update(row)
            .eq("id", user["profile_id"])
            .execute()
        )

        return (
            res.data[0]
            if res.data
            else row
        )

    res = (
        supabase
        .table("user_profiles")
        .insert(row)
        .execute()
    )

    profile = res.data[0]

    (
        supabase
        .table("users")
        .update({
            "profile_id": profile["id"]
        })
        .eq("id", user["id"])
        .execute()
    )

    return profile
def update_user_memory(user_code: str, new_data: dict):
    user = get_or_create_user(user_code)
    current = get_user_memory(user_code)

    allowed_fields = {
        "health_interests",
        "last_detected_issue",
        "last_recommended_product",
        "past_recommended_products",
        "recurring_food_patterns",
        "recurring_activity_patterns",
        "last_meal_summary",
        "last_activity_summary",
        "consistency_score",
        "notes",
        "trend_analysis",
    }

    list_fields = {
        "health_interests",
        "past_recommended_products",
        "recurring_food_patterns",
        "recurring_activity_patterns",
        "notes",
    }

    update_data = {}

    for key, value in new_data.items():
        if key not in allowed_fields:
            continue

        if value is None or value == "":
            continue

        if key in list_fields:
            update_data[key] = merge_unique(
                current.get(key),
                value
            )
        else:
            update_data[key] = value

    if not update_data:
        return

    (
        supabase.table("user_memory")
        .update(update_data)
        .eq("user_id", user["id"])
        .execute()
    )