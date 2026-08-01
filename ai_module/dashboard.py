import json
from datetime import date, timedelta
from openai import OpenAI
import os

from ai_module.user_memory_db import (
    get_user_profile,
    get_last_days_checkins,
    get_checkins_between,
    get_weight_history,
    get_loyalty,
)
from ai_module.product_db import (
    get_product_knowledge_dict,
)


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


LOYALTY_LEVELS = [
    {
        "name": "Bronze",
        "minimum_points": 0,
    },
    {
        "name": "Silver",
        "minimum_points": 300,
    },
    {
        "name": "Gold",
        "minimum_points": 600,
    },
    {
        "name": "Platinum",
        "minimum_points": 1000,
    },
]

def calculate_weight_goal_progress(
    starting_weight,
    current_weight,
    target_weight,
) -> dict:
    if any(
        value is None
        for value in [
            starting_weight,
            current_weight,
            target_weight,
        ]
    ):
        return {
            "starting_weight": starting_weight,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "progress_percent": None,
            "remaining_kg": None,
            "status": "missing_data",
        }

    starting_weight = float(starting_weight)
    current_weight = float(current_weight)
    target_weight = float(target_weight)

    # No change is required
    if starting_weight == target_weight:
        return {
            "starting_weight": starting_weight,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "progress_percent": 100,
            "remaining_kg": 0,
            "status": "completed",
        }

    # Weight-loss objective
    if target_weight < starting_weight:
        total_needed = starting_weight - target_weight
        achieved = starting_weight - current_weight

        progress = (achieved / total_needed) * 100
        remaining = max(current_weight - target_weight, 0)

        completed = current_weight <= target_weight

    # Weight-gain objective
    else:
        total_needed = target_weight - starting_weight
        achieved = current_weight - starting_weight

        progress = (achieved / total_needed) * 100
        remaining = max(target_weight - current_weight, 0)

        completed = current_weight >= target_weight

    progress = round(
        max(0, min(progress, 100)),
        1,
    )

    remaining = round(remaining, 1)

    if completed:
        status = "completed"
    elif progress <= 0:
        status = "started"
    elif progress >= 75:
        status = "almost_there"
    else:
        status = "in_progress"

    return {
        "starting_weight": starting_weight,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "progress_percent": progress,
        "remaining_kg": remaining,
        "status": status,
    }

def calculate_loyalty_progress(
    points: int,
) -> dict:
    points = int(points or 0)

    current_level = (
        LOYALTY_LEVELS[0]
    )

    next_level = None

    for index, level in enumerate(
        LOYALTY_LEVELS
    ):
        if points >= level["minimum_points"]:
            current_level = level

            if index + 1 < len(
                LOYALTY_LEVELS
            ):
                next_level = (
                    LOYALTY_LEVELS[index + 1]
                )

    if next_level is None:
        return {
            "points": points,
            "current_level": (
                current_level["name"]
            ),
            "next_level": None,
            "points_to_next_level": 0,
            "progress_percent": 100,
        }

    current_minimum = current_level[
        "minimum_points"
    ]

    next_minimum = next_level[
        "minimum_points"
    ]

    interval = (
        next_minimum - current_minimum
    )

    progress = round(
        (
            (points - current_minimum)
            / interval
        )
        * 100,
        1,
    )

    return {
        "points": points,
        "current_level": (
            current_level["name"]
        ),
        "next_level": next_level["name"],
        "points_to_next_level": max(
            next_minimum - points,
            0,
        ),
        "progress_percent": max(
            0,
            min(progress, 100),
        ),
    }

def normalize_loyalty_language(language: str | None) -> str:
    language = str(language or "ar").lower().strip()

    if language.startswith("fr"):
        return "fr"

    if language.startswith("en"):
        return "en"

    return "ar"

def build_loyalty_message_text(
    language: str,
    product_name: str,
    reward: int,
    before: dict,
    after: dict,
    points_to_promotion: int,
    points_to_free_product: int,
) -> str:
    language = normalize_loyalty_language(language)

    current_level = before.get("current_level")
    level_after_purchase = after.get("current_level")

    next_level_after = after.get("next_level")
    points_to_next_after = after.get(
        "points_to_next_level",
        0,
    )

    points_after_purchase = after.get(
        "points",
        0,
    )

    reaches_new_level = (
        level_after_purchase != current_level
    )

    # French
    if language == "fr":
        if next_level_after is None:
            if reaches_new_level:
                return (
                  
                    f"Achetez {product_name} aujourd’hui et gagnez {reward} points de fidélité ! "
                    f"Plus que {points_to_promotion} points pour débloquer votre prochaine offre exclusive "
                    f"et {points_to_free_product} points pour obtenir un produit gratuit."
                )

            return (
                f"Vous êtes déjà au niveau de fidélité le plus élevé. "
                f"L’achat de {product_name} vous rapporte {reward} points supplémentaires."
            )

        if reaches_new_level:
            return (
                f"L’achat de {product_name} vous rapporte {reward} points "
                f"et vous permet d’atteindre le niveau {level_after_purchase}. "
                f"Après l’achat, vous aurez {points_after_purchase} points "
                f"et il vous manquera {points_to_next_after} points "
                f"pour atteindre le niveau {next_level_after}."
            )

        return (
            f"L’achat de {product_name} vous rapporte {reward} points. "
            f"Après l’achat, vous aurez {points_after_purchase} points "
            f"et il vous manquera {points_to_next_after} points "
            f"pour atteindre le niveau {next_level_after}."
        )

    # English
    if language == "en":
        if next_level_after is None:
            if reaches_new_level:
                return (
                    
                    f"Buy {product_name} today and earn {reward} loyalty points! "
                    f"Only {points_to_promotion} more points to unlock your next exclusive offer "
                    f"and {points_to_free_product} points to claim a free product."
                )

            return (
                f"You are already at the highest loyalty level. "
                f"Buying {product_name} gives you {reward} additional points."
            )

        if reaches_new_level:
            return (
                f"Buying {product_name} gives you {reward} points "
                f"and moves you to {level_after_purchase}. "
                f"After the purchase, you will have {points_after_purchase} points "
                f"and need {points_to_next_after} more points "
                f"to reach {next_level_after}."
            )

        return (
            f"Buying {product_name} gives you {reward} points. "
            f"After the purchase, you will have {points_after_purchase} points "
            f"and need {points_to_next_after} more points "
            f"to reach {next_level_after}."
        )

    # Tunisian Arabic
    if next_level_after is None:
        if reaches_new_level:
            return (
                f"اشترِ {product_name} اليوم واربح {reward} نقطة ولاء! "
                f"باقي لك {points_to_promotion} نقطة فقط لفتح العرض القادم، "
                f"و{points_to_free_product} نقطة للحصول على منتج مجاني."
            )

        return (
            f"إنتي وصلت لأعلى مستوى في برنامج الولاء. "
            f"شراء {product_name} يعطيك {reward} نقطة إضافية."
        )

    if reaches_new_level:
        return (
            f"شراء {product_name} يعطيك {reward} نقطة "
            f"ويطلعك لمستوى {level_after_purchase}. "
            f"بعد الشراء باش يكون عندك {points_after_purchase} نقطة، "
            f"ويبقالك {points_to_next_after} نقطة باش توصل "
            f"لمستوى {next_level_after}."
        )

    return (
        f"استفد من عرض {product_name} واربح {reward} نقطة ولاء. "
        f"بعد الاستفادة من العرض باش يكون عندك {points_after_purchase} نقطة، "
        f"وباقيلك {points_to_next_after} نقطة فقط باش توصل "
        f"لمستوى {next_level_after}."
    )

def build_product_loyalty_message(
    user_code: str,
    product_name: str | None,
    language: str = "ar",
) -> dict | None:
    if not user_code or not product_name:
        return None

    products = (
        get_product_knowledge_dict()
    )

    product = products.get(product_name)

    if not product:
        return None
    print("USER CODE =", user_code)
    loyalty_row = get_loyalty(user_code) or {}

    current_points = int(
        loyalty_row.get("points") or 0
    )

    reward = int(
        product.get(
            "loyalty_points_reward"
        )
        or 0
    )

    before = calculate_loyalty_progress(
        current_points
    )

    after = calculate_loyalty_progress(
        current_points + reward
    )

    points_after_purchase = current_points + reward

    promotion_target = 1000
    free_product_target = 8000

    points_to_promotion = max(
        promotion_target - points_after_purchase,
        0,
    )

    points_to_free_product = max(
        free_product_target - points_after_purchase,
        0,
    )

    return {
        "product": product_name,
        "current_points": current_points,
        "reward_points": reward,
        "points_after_purchase": current_points + reward,

        "current_level": before["current_level"],

        # After buying, what level are we heading toward?
        "next_level": after["next_level"],

    # Remaining points after the reward has been added
        "points_to_next_level": after["points_to_next_level"],

        "level_after_purchase": after["current_level"],
        "points_to_promotion": points_to_promotion,
        "points_to_free_product": points_to_free_product,
        "message": build_loyalty_message_text(
            language=language,
            product_name=product_name,
            reward=reward,
            before=before,
            after=after,
            points_to_promotion=points_to_promotion,
            points_to_free_product=points_to_free_product,
        ),
    }
def percentage_change(
    current: float,
    previous: float,
) -> float | None:
    if previous == 0:
        return None

    return round(
        ((current - previous) / previous)
        * 100,
        1,
    )

def build_weekly_metrics(
    current_week: list[dict],
    previous_week: list[dict],
) -> dict:
    current_activity = sum(
        float(
            item.get(
                "activity_calories_burned"
            )
            or 0
        )
        for item in current_week
    )

    previous_activity = sum(
        float(
            item.get(
                "activity_calories_burned"
            )
            or 0
        )
        for item in previous_week
    )

    current_calories = sum(
        float(
            item.get("calories_in")
            or 0
        )
        for item in current_week
    )

    previous_calories = sum(
        float(
            item.get("calories_in")
            or 0
        )
        for item in previous_week
    )

    current_sugar_days = sum(
        1
        for item in current_week
        if "high_sugar" in (
            item.get("food_patterns")
            or []
        )
    )

    previous_sugar_days = sum(
        1
        for item in previous_week
        if "high_sugar" in (
            item.get("food_patterns")
            or []
        )
    )

    return {
        "current_checkin_days": len(
            current_week
        ),
        "previous_checkin_days": len(
            previous_week
        ),
        "activity_change_percent": (
            percentage_change(
                current_activity,
                previous_activity,
            )
        ),
        "calories_change_percent": (
            percentage_change(
                current_calories,
                previous_calories,
            )
        ),
        "current_sugar_days": (
            current_sugar_days
        ),
        "previous_sugar_days": (
            previous_sugar_days
        ),
        "sugar_days_change": (
            current_sugar_days
            - previous_sugar_days
        ),
    }

def generate_weekly_ai_summary(
    metrics: dict,
    language: str = "ar",
) -> str:
    prompt = f"""
Create a short personalized weekly wellness summary.

Language: {language}

Calculated metrics:
{json.dumps(metrics, ensure_ascii=False)}

Rules:
- Use only the supplied metrics.
- Maximum 3 short sentences.
- Mention one improvement when available.
- Mention an improvement opportunity only when the supplied metrics clearly support it.
- A value of zero for sugar days means no high-sugar days were detected; do not say the user failed to track sugar.
- If previous-week data is missing, do not claim improvement or decline compared with last week.
- Be motivating.
- Do not invent percentages.
- Do not diagnose.
"""

    try:
        response = (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Use only the calculated "
                            "metrics supplied."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )
        )

        return (
            response.choices[0]
            .message.content
            .strip()
        )

    except Exception:
        return (
            "Your weekly summary is temporarily "
            "unavailable."
        )
    
def build_graph_data(
    checkins: list[dict],
    weight_history: list[dict],
) -> dict:
    return {
        "calories": [
            {
                "date": row.get(
                    "checkin_date"
                ),
                "calories_in": row.get(
                    "calories_in"
                ),
                "calories_burned": row.get(
                    "total_calories_burned"
                ),
                "net_calories": row.get(
                    "net_calories"
                ),
            }
            for row in checkins
        ],
        "activity": [
            {
                "date": row.get(
                    "checkin_date"
                ),
                "activity_minutes": row.get(
                    "activity_minutes"
                ),
                "activity_calories": row.get(
                    "activity_calories_burned"
                ),
                "steps": row.get(
                    "daily_steps"
                ),
            }
            for row in checkins
        ],
        "weight": [
            {
                "date": row.get(
                    "logged_at"
                ),
                "weight": row.get(
                    "weight"
                ),
            }
            for row in weight_history
        ],
        "consistency": [
            {
                "date": row.get(
                    "checkin_date"
                ),
                "consistency_score": row.get(
                    "consistency_score"
                ),
            }
            for row in checkins
        ],
        "checkins": [
            {
                "date": row.get(
                    "checkin_date"
                ),
                "checkin_count": row.get(
                    "checkin_count"
                ),
            }
            for row in checkins
        ],
        "weight_trend": [
            {
                "date": row.get(
                    "checkin_date"
                ),
                "estimated_weekly_change_kg": (
                    row.get(
                        "estimated_weekly_weight_change_kg"
                    )
                ),
                "trend": row.get(
                    "weight_trend"
                ),
            }
            for row in checkins
        ],
    }

def build_dashboard(
    user_code: str,
) -> dict:
    profile = get_user_profile(user_code)

    current_week = get_last_days_checkins(
        user_code,
        7,
    )

    today = date.today()

    previous_end = (
        today - timedelta(days=7)
    )

    previous_start = (
        today - timedelta(days=13)
    )

    previous_week = get_checkins_between(
        user_code,
        str(previous_start),
        str(previous_end),
    )

    weekly_metrics = build_weekly_metrics(
        current_week,
        previous_week,
    )

    weekly_summary = (
        generate_weekly_ai_summary(
            weekly_metrics,
            profile.get("language") or "ar",
        )
    )

    weight_history = get_weight_history(user_code)

    latest_weight = profile.get("weight")

    if weight_history:
        latest_weight = weight_history[-1]["weight"]

    starting_weight = (
        profile.get("starting_weight")
        or profile.get("weight")
    )

    current_weight = latest_weight
    target_weight = profile.get("target_weight")

    weight_progress = (
        calculate_weight_goal_progress(
            starting_weight,
            current_weight,
            target_weight,
        )
    )
    loyalty_row = get_loyalty(user_code)

    loyalty_progress = (
        calculate_loyalty_progress(
            loyalty_row.get("points", 0)
        )
    )


    return {
        "weight_goal_progress": (
            weight_progress
        ),
        "weekly_metrics": weekly_metrics,
        "weekly_ai_summary": weekly_summary,
        "loyalty": loyalty_progress,
        "graphs": build_graph_data(
            current_week,
            weight_history,
        ),
    }

