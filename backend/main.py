from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date


from ai_module.checkin_agent import (
    build_daily_checkin_output,
)

from ai_module.user_memory_db import (
    create_daily_checkin,
    log_daily_meal,
    log_daily_activity,
    update_daily_summary,
    update_user_memory,
    add_loyalty_points,
    log_weight,
    upsert_user_profile,
)

from ai_module.dashboard import (
    build_dashboard,
    build_product_loyalty_message,
)
from ai_module.Chatbot import chatbot_response

from ai_module.recommendation_agent import build_recommendation_agent_output
from ai_module.user_memory_db import (
    
    get_recent_checkins,
)
app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    user_id: str | None = None
    age: int | None = None
    height: float | None = None
    weight: float | None = None
    sex: str | None = None
    goals: list[str] | None = None
    preferences: str | None = None
    activity_info: str | None = None
    history: str | None = None
    chat_history: Optional[List[Dict[str, str]]] = None
    medical_conditions: list[str] | None = None
    language: str = "ar"





class RecommendRequest(BaseModel):
    user_id: str
    age: int | None = None
    weight: float | None = None
    height: float | None = None
    sex: str | None = None
    goals: list[str] | None = None
    language: str = "ar"
    activity_info: str | None = None
    medical_conditions: list[str] | None = None
class CheckinRequest(BaseModel):
    user_id: str
    meals_today: str = ""
    activity_today: str = ""
    mood: str = ""
    age: int | None = None
    weight: float | None = None
    height: float | None = None

    starting_weight: float | None = None
    target_weight: float | None = None

    sex: str | None = None
    goals: list[str] = []
    medical_conditions: list[str] = []
    language: str = "ar"
    daily_steps: int = 0
class WeightRequest(BaseModel):
    user_id: str
    weight: float

@app.get("/")
def root():
    return {"message": "ShifaChatbot API is running"}


@app.post("/chat")
def chat(req: ChatRequest):
    user_profile = {
        "user_id": req.user_id or "demo_user",
        "age": req.age,
        "weight": req.weight,
        "height": req.height,
        "sex": req.sex,
        "goals": req.goals or [],
        "preferences": req.preferences,
        "activity_info": req.activity_info,
        "history": req.history,
        "medical_conditions": req.medical_conditions or [],
        "language": req.language,
    }

    result = chatbot_response(
        question=req.question,
        user_profile=user_profile,
        chat_history=req.chat_history or []
    )

    return {
        "response": result.get("answer"),
        "intent": result.get("intent"),
        "detected_product": result.get(
            "detected_product"
        ),
        "recommended_product": result.get(
            "recommended_product"
        ),
        "recommendation_reason": result.get(
            "recommendation_reason"
        ),
        "meal_suggestion": result.get(
            "meal_suggestion"
        ),
        "calorie_info": result.get(
            "calorie_info"
        ),
        "usage_info": result.get(
            "usage_info"
        ),
        "benefits_info": result.get(
            "benefits_info"
        ),
        "precautions": result.get(
            "precautions"
        ),
        "lifestyle_suggestion": result.get(
            "lifestyle_suggestion"
        ),
        "loyalty_recommendation": result.get(
            "loyalty_recommendation"
        ),
        "follow_up_question": result.get(
            "follow_up_question"
        ),
        "price_info": result.get(
            "price_info"
        ),
        "offer_info": result.get(
            "offer_info"
        ),
    }


@app.post("/checkin")
def create_checkin(request: CheckinRequest):
    user_profile = {
        "user_id": request.user_id,
        "age": request.age,
        "weight": request.weight,
        "height": request.height,
        "sex": request.sex,
        "goals": request.goals,
        "medical_conditions": (
            request.medical_conditions
        ),
        "language": request.language,
    }





    profile_data = {
        "age": request.age,
        "height": request.height,
        "weight": request.weight,
        "starting_weight": request.starting_weight,
        "target_weight": request.target_weight,
        "sex": request.sex,
        "goals": request.goals,
        "language": request.language,
        "medical_conditions": request.medical_conditions,
    }

    profile_data = {
        key: value
        for key, value in profile_data.items()
        if value not in (None, "", "string")
    }

    upsert_user_profile(
        request.user_id,
        profile_data,
    )

    current_result = (
        build_daily_checkin_output(
            user_profile=user_profile,
            meals_today=request.meals_today,
            activity_today=(
                request.activity_today
            ),
            mood=request.mood,
            steps=request.daily_steps,
        )
    )

    energy = current_result.get(
        "energy_estimation",
        {},
    )

    structured = current_result.get(
        "structured_energy",
        {},
    )

    checkin = create_daily_checkin(
        request.user_id,
        {
            "checkin_date": str(
                date.today()
            ),
            "mood": request.mood,
            "daily_steps": request.daily_steps,
            "consistency_score": current_result.get(
                "consistency_score",
                0,
            ),
        },
    )

    if request.meals_today.strip():
        log_daily_meal(
            checkin["id"],
            {
                "meal_type": "general",
                "description": (
                    request.meals_today
                ),
                "estimated_calories": (
                    energy.get(
                        "estimated_calories_in"
                    )
                ),
            },
        )

    if request.activity_today.strip():

           

        activity_text = (
            request.activity_today
            or ""
        ).strip().lower()

        no_activity_values = {
            "",
            "no activity",
            "none",
            "nothing",
            "pas d'activité",
            "pas d activite",
            "hata chay",
            "7ata chay",
            "حتى شيء",
            "لا شيء",
        }

        if activity_text not in no_activity_values:
            log_daily_activity(
                checkin["id"],
                {
                    "activity_type": request.activity_today,
                    "description": request.activity_today,
                    "duration_minutes": structured.get(
                        "activity_duration_minutes"
                    ),
                    "intensity": structured.get(
                        "activity_intensity"
                    ),
                    "estimated_calories_burned": structured.get(
                        "estimated_activity_calories_burned"
                    ),
                },
            )

    memory_updates = current_result.get(
        "memory_updates",
        {},
    )

    update_user_memory(
        request.user_id,
        memory_updates,
    )

    daily_summary = update_daily_summary(
        request.user_id,
        user_profile,
        str(date.today()),
    )

    loyalty = add_loyalty_points(
        request.user_id,
        10,
    )

    return {
        "current_checkin": current_result,
        "daily_summary": daily_summary,
        "loyalty_points": loyalty.get(
            "points",
            0,
        ),
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    user_profile = {
        "user_id": req.user_id,
        "age": req.age,
        "weight": req.weight,
        "height": req.height,
        "sex": req.sex,
        "goals": req.goals or [],
        "language": req.language,
        "activity_info": req.activity_info,
        "medical_conditions": req.medical_conditions or [],
    }

    return build_recommendation_agent_output(user_profile)

@app.get("/dashboard/{user_id}")
def dashboard(user_id: str):
    return build_dashboard(user_id)
@app.post("/weight")
def save_weight(request: WeightRequest):
    saved = log_weight(
        request.user_id,
        request.weight,
    )

    return {
        "message": "Weight saved.",
        "weight_log": saved,
    }