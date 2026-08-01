import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import date
from ai_module.dashboard import build_product_loyalty_message

from ai_module.product_db import (
    get_product_knowledge_dict,
    get_quantity_offers_dict,
    get_bundle_offers,
    
)  
from ai_module.user_memory_db import (
    get_user_memory,
    get_user_profile,
    log_recommendation,
    get_recent_checkins,
    get_recent_chat_history
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "fr": "French",
    "en": "English",
}
PRODUCT_SIGNAL_ALIASES = {
    "Colon Detox": {
        "digestion",
        "bloating_gas",
        "bloating",
        "gas",
        "digestive health",
        "colon",
        "gut",
        "constipation",
    },

    "Psyllium": {
        "constipation",
        "fiber_support",
        "fiber",
        "fibre" ,
        "bowel movement",
        "bowel regularity",
        "satiety",
        "appetite_control",
        "weight_loss",
        "weight management",
    },

    "Liver Detox": {
        "detox",
        "liver",
        "liver support",
    },

    "Slim Pack": {
        "weight_loss",
        "weight management",
        "slimming",
        "appetite",
        "metabolism",
    },

    "Blood Detox": {
        "blood_regulation",
        "heart_support",
        "circulation",
        "heart",
    },

    "Lung Detox": {
        "respiratory_support",
        "smoking_reduction_support",
        "lungs",
        "lung support",
        "breathing",
        "smoking",
        "respiratory",
    },

    "Berberine & Ceylon Cinnamon": {
        "glycemic_balance",
        "insulin_sensitivity",
        "sugar_cravings",
        "blood sugar",
        "glucose",
        "insulin",
   
    },
}

def normalize_language(language: str | None) -> str:
    language = str(language or "ar").lower().strip()
    return language if language in LANGUAGE_NAMES else "ar"


def translated_text(language: str, key: str) -> str:
    language = normalize_language(language)

    texts = {
        "ar": {
            "supplement_warning": (
                "منتجات شفاء مكملات غذائية وليست أدوية، ولا تعوّض استشارة الطبيب."
            ),
            "persistent_symptoms_warning": (
                "إذا كانت الأعراض متكررة أو قوية، من الأفضل استشارة مختص."
            ),
            "chronic_condition_warning": (
                "إذا كان لديك مرض مزمن أو تستعمل أدوية، استشر طبيباً أو صيدلياً "
                "قبل استعمال أي مكمل غذائي."
            ),
            "pregnancy_warning": (
                "خلال فترة الحمل أو الرضاعة، استشيري الطبيب قبل استعمال أي مكمل غذائي."
            ),
            "best_offer": "هذا أفضل عرض متوفر حالياً لهذا المنتج.",
            "small_habits_action": "ابدأ بعادات صغيرة وسهلة التطبيق.",
            "small_habits_reason": (
                "التغييرات التدريجية تساعدك على تحسين الاستمرارية على المدى الطويل."
            ),
            "movement_action": "زد مستوى الحركة تدريجياً.",
            "movement_reason": (
                "البدء تدريجياً يساعدك على تحسين نشاطك بطريقة واقعية وآمنة."
            ),
            "stress_action": "حاول تحسين النوم والتقليل من التوتر.",
            "stress_reason": (
                "إدارة التوتر والنوم الجيد يدعمان التعافي والصحة العامة."
            ),
            "colon_reason": (
                "تم اختياره لدعم الهضم والتخفيف من النفخة والغازات والإمساك."
            ),
            "liver_reason": (
                "تم اختياره لدعم وظائف الكبد وروتين التخلص الطبيعي من السموم."
            ),
            "slim_reason": (
                "تم اختياره ليتناسب مع هدف التحكم في الوزن."
            ),
            "blood_reason": (
                "تم اختياره لدعم الدورة الدموية وصحة القلب والراحة العامة."
            ),
            "fallback_reasoning": (
                "حضّرت لك هذه الخطة حسب هدفك وملفك وعاداتك الأخيرة "
                "حتى تساعدك على التقدم بطريقة صحية وتدريجية."
            ),
            "lung_reason": (
                "اخترنا هذا المنتج لدعم صحة الرئتين والراحة أثناء التنفس، "
                "خصوصاً ضمن خطة تدريجية للتقليل من التدخين."
            ),
            "berberine_reason": (
                "اخترنا هذا المنتج لدعم التوازن الطبيعي للسكر، "
                "وحساسية الإنسولين والتحكم في الرغبة في السكريات."
            ),
            "psyllium_reason": (
                "اخترنا هذا المنتج لدعم الهضم، وانتظام حركة الأمعاء، "
                "وزيادة الشعور بالشبع."
            ),

            "diabetes_warning": (
                "إذا كنت مصابًا بالسكري أو تستعمل أدوية لتنظيم السكر، "
                "استشر طبيبك قبل استعمال هذا المكمل. "
                    "هذا المنتج لا يُعتبر بديلاً عن العلاج الطبي."
                ),

            "hypertension_warning": (
                "إذا كنت تعاني من ارتفاع ضغط الدم أو تستعمل أدوية للضغط، "
                "استشر طبيبك قبل استعمال هذا المكمل. "
                "هذا المنتج لا يُعتبر بديلاً عن العلاج الطبي."
            ),
        },

        "fr": {
            "supplement_warning": (
                "Les produits Shifa sont des compléments alimentaires et ne remplacent "
                "pas un avis médical."
            ),
            "persistent_symptoms_warning": (
                "Si les symptômes sont fréquents ou importants, il est préférable "
                "de consulter un professionnel de santé."
            ),
            "chronic_condition_warning": (
                "Si vous souffrez d’une maladie chronique ou prenez des médicaments, "
                "consultez un médecin ou un pharmacien avant d’utiliser un complément."
            ),
            "pregnancy_warning": (
                "Pendant la grossesse ou l’allaitement, consultez votre médecin "
                "avant d’utiliser un complément alimentaire."
            ),
            "best_offer": "Voici la meilleure offre actuellement disponible pour ce produit.",
            "small_habits_action": "Commencez par de petites habitudes faciles à maintenir.",
            "small_habits_reason": (
                "Des changements progressifs favorisent une meilleure régularité "
                "sur le long terme."
            ),
            "movement_action": "Augmentez progressivement votre niveau d’activité.",
            "movement_reason": (
                "Une progression graduelle permet d’améliorer votre activité "
                "de manière réaliste et sûre."
            ),
            "stress_action": "Améliorez votre sommeil et réduisez votre stress.",
            "stress_reason": (
                "Une meilleure gestion du stress favorise la récupération "
                "et le bien-être général."
            ),
            "colon_reason": (
                "Sélectionné pour soutenir la digestion et aider en cas de ballonnements, "
                "gaz ou constipation."
            ),
            "liver_reason": (
                "Sélectionné pour soutenir le fonctionnement du foie "
                "et la détoxification naturelle."
            ),
            "slim_reason": (
                "Sélectionné pour accompagner votre objectif de gestion du poids."
            ),
            "blood_reason": (
                "Sélectionné pour soutenir la circulation sanguine, le cœur "
                "et le bien-être général."
            ),
            "fallback_reasoning": (
                "J’ai préparé ce plan selon votre objectif, votre profil et vos habitudes "
                "récentes afin de vous aider à progresser sainement."
            ),
            "lung_reason": (
                "Sélectionné pour soutenir le confort respiratoire et la santé des poumons, "
                "notamment lors d’une réduction progressive du tabac."
            ),
            "berberine_reason": (
                "Sélectionné pour soutenir l’équilibre glycémique normal, "
                "la sensibilité à l’insuline et le contrôle des envies de sucre."
            ),
            "psyllium_reason": (
                "Sélectionné pour soutenir le transit intestinal, "
                "le confort digestif et la satiété."
            ),

            "diabetes_warning": (
                "Si vous êtes diabétique ou sous traitement pour le diabète, "
                "consultez un professionnel de santé avant utilisation. "
                "Ce complément ne remplace pas votre traitement médical."
            ),

            "hypertension_warning": (
                "Si vous souffrez d'hypertension ou prenez un traitement antihypertenseur, "
                "consultez un professionnel de santé avant utilisation. "
                "Ce complément ne remplace pas votre traitement médical."
            ),
        },

        "en": {
            "supplement_warning": (
                "Shifa products are dietary supplements and do not replace medical advice."
            ),
            "persistent_symptoms_warning": (
                "If symptoms are frequent or severe, consult a healthcare professional."
            ),
            "chronic_condition_warning": (
                "If you have a chronic condition or take medication, consult a doctor "
                "or pharmacist before using any supplement."
            ),
            "pregnancy_warning": (
                "During pregnancy or breastfeeding, consult your doctor before using "
                "any dietary supplement."
            ),
            "best_offer": "This is the best currently available offer for this product.",
            "small_habits_action": "Start with small, achievable habits.",
            "small_habits_reason": (
                "Gradual changes improve long-term consistency."
            ),
            "movement_action": "Increase your movement progressively.",
            "movement_reason": (
                "A gradual approach helps improve activity safely and realistically."
            ),
            "stress_action": "Improve your sleep and reduce stress.",
            "stress_reason": (
                "Stress management and quality sleep support recovery and wellbeing."
            ),
            "colon_reason": (
                "Selected to support digestion and help with bloating, gas, "
                "and constipation."
            ),
            "liver_reason": (
                "Selected to support liver function and the body’s natural detox routine."
            ),
            "slim_reason": (
                "Selected to support your weight-management goal."
            ),
            "blood_reason": (
                "Selected to support circulation, heart health, and general wellbeing."
            ),
            "fallback_reasoning": (
                "I prepared this plan using your goal, profile, and recent habits "
                "to help you progress in a healthy and gradual way."
            ),
            "lung_reason": (
                "Selected to support lung wellness and comfortable breathing, "
                "including during gradual smoking reduction."
            ),
            "berberine_reason": (
                "Selected to support normal glucose balance, insulin sensitivity, "
                "and management of sugar cravings."
            ),
            "psyllium_reason": (
                "Selected to support digestion, bowel regularity, fiber intake, "
                "and satiety."
            ),

            "diabetes_warning": (
                "If you have diabetes or take medication for blood sugar control, "
                "consult your healthcare professional before use. "
                "This supplement does not replace medical treatment."
            ),

            "hypertension_warning": (
                "If you have high blood pressure or take medication for hypertension, "
                "consult your healthcare professional before use. "
                "This supplement does not replace medical treatment."
    ),
        },
    }

    return texts[language].get(key, "")

def calculate_bmi(weight, height_cm):
    if not weight or not height_cm:
        return None
    height_m = height_cm / 100
    if height_m <= 0:
        return None
    return round(weight / (height_m ** 2), 1)


def get_best_offer(
    product_name,
    quantity_offers_db,
    bundle_offers_db,
):
    offers = quantity_offers_db.get(product_name, [])

    active_offers = [
        offer
        for offer in offers
        if offer.get("new_price") is not None
    ]

    if active_offers:
        return max(
            active_offers,
            key=lambda offer: (
                offer.get("discount_percent", 0),
                -offer.get("quantity", 1),
            ),
        )

    for bundle in bundle_offers_db:
        if (
            bundle.get("active", True)
            and product_name in bundle.get("products", [])
        ):
            return bundle

    return None


def extract_recommendation_signals_with_llm(merged_profile: dict) -> dict:
    fallback = {
        "health_interests": merged_profile.get("health_interests", []),
        "activity_level": "unknown",
        "food_patterns": merged_profile.get("recurring_food_patterns", []),
        "detected_priority_need": None,
        "recommended_product_signals": [],
        "meal_strategy": "",
        "exercise_strategy": "",
        "reasoning_summary": "Fallback extraction used."
    }

    if client is None:
        return fallback
    prompt = f"""
You are a wellness recommendation signal extractor for Shifa.

Analyze this user profile + memory + daily check-in data and return ONLY valid JSON.
Also use recent chatbot interactions from the last 90 days, but do not treat the full conversation as equally important.
Use only useful signals such as health concerns, product questions, repeated issues, preferences, and previous recommendations.
Ignore greetings, thanks, small talk, and unrelated messages.

Give importance in this order:
1. Last 7 days check-in/journal behavior
2. User profile and goals
3. Last 90 days useful chatbot interactions
4. Long-term memory

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

Allowed activity_level:
- low
- moderate
- high
- unknown

Allowed food_patterns:
- heavy_meals
- high_sugar
- light_meals
- balanced_meals
- protein_intake
- vegetable_intake

Return this exact JSON schema:
{{
  "health_interests": [],
  "activity_level": "unknown",
  "food_patterns": [],
  "detected_priority_need": null,
  "recommended_product_signals": [],
  "meal_strategy": "",
  "exercise_strategy": "",
  "reasoning_summary": ""
}}

Product mapping signals:
- gas, bloating, general digestive discomfort -> Colon Detox
- constipation, low fiber, bowel regularity, satiety -> Psyllium
- liver or detox interest -> Liver Detox
- weight loss or high BMI -> Slim Pack
- lung wellness, breathing comfort, smoking reduction -> Lung Detox
- glucose balance, insulin sensitivity, sugar cravings -> Berberine & Ceylon Cinnamon
- blood circulation or heart support -> Blood Detox
- muscle gain -> no product unless the product database supports it

Safety:
- Do not describe any product as treating or curing disease.
- Berberine & Ceylon Cinnamon does not replace diabetes treatment.
- Lung Detox does not guarantee smoking cessation.
- Pregnancy, breastfeeding, chronic conditions, and medication use require professional advice.

Important:
- Do not invent products.
- Use only signals, not final long explanations.
- If anxiety or stress is mentioned, use stress_anxiety only.
- Use blood_regulation only when the user explicitly mentions circulation,
  heart health, blood pressure, or another cardiovascular concern.

User data:
{json.dumps(merged_profile, ensure_ascii=False)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown. No explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception:
        return fallback


def add_product(recommended_products, product, reason, quantity_offers_db, bundle_offers_db):
    if product not in [p["product"] for p in recommended_products]:
        recommended_products.append({
            "product": product,
            "reason": reason,
            "offer": get_best_offer(product, quantity_offers_db, bundle_offers_db)
        })
def normalize_product_tags(product: dict) -> set[str]:
    tags = product.get("tags") or []

    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",")]

    return {
        str(tag).lower().strip()
        for tag in tags
        if str(tag).strip()
    }


def build_user_product_signals(
    signals: dict,
    merged_profile: dict,
    bmi: float | None,
) -> set[str]:
    user_signals = set()

    for item in signals.get("health_interests", []) or []:
        user_signals.add(str(item).lower().strip())

    for item in signals.get("food_patterns", []) or []:
        user_signals.add(str(item).lower().strip())

    priority = signals.get("detected_priority_need")
    if priority:
        user_signals.add(str(priority).lower().strip())

    for item in signals.get("recommended_product_signals", []) or []:
        user_signals.add(str(item).lower().strip())

    goals = merged_profile.get("goals") or []
    if not isinstance(goals, list):
        goals = [goals]

    for goal in goals:
        user_signals.add(str(goal).lower().strip())

    conditions = merged_profile.get("medical_conditions") or []
    if not isinstance(conditions, list):
        conditions = [conditions]

    conditions_text = " ".join(
        str(condition).lower()
        for condition in conditions
    )


    if "hypertension" in conditions_text:
        user_signals.update({
            "heart_support",
            "circulation",
        })
    
    if "diabetes" in conditions_text:
        user_signals.update({
            "glycemic_balance",
            "blood sugar",
            "insulin_sensitivity",
        })

    if bmi and bmi >= 25:
        user_signals.update({
            "weight_loss",
            "weight management",
        })

    if "diabetes" in conditions_text:
        user_signals.update({
            "glycemic_balance",
            "blood sugar",
        })

    return user_signals
def normalize_moods(recent_moods):
    moods_text = " ".join([str(m).lower() for m in recent_moods])

    mood_map = {
        "😔": "low_mood stress low_energy",
        "😐": "neutral",
        "🙂": "good relaxed",
        "😊": "motivated energetic",
        "🤩": "highly_motivated energetic"
    }

    for emoji, meaning in mood_map.items():
        if emoji in moods_text:
            moods_text += " " + meaning

    return moods_text
def build_nutrition_strategy(merged_profile, signals):
    strategy = {
        "calorie_approach": "balanced",
        "protein": "normal",
        "fiber": "normal",
        
        "step_goal": "",
        "avoid": [],
        "focus": []
    }

    goals = merged_profile.get("goals", [])
    bmi = merged_profile.get("bmi")
    food_patterns = signals.get("food_patterns", [])
    health = signals.get("health_interests", [])
    conditions = merged_profile.get("medical_conditions", [])
    conditions_text = " ".join(conditions).lower() if isinstance(conditions, list) else str(conditions).lower()
    sex = merged_profile.get("sex")
    age = merged_profile.get("age")
   
    recent_moods = merged_profile.get("recent_moods", [])
    avg_steps = merged_profile.get("average_steps_last_7_days")
    activity_level = signals.get("activity_level", "unknown")
   
    # Weight loss
    if "weight_loss" in goals or (bmi and bmi >= 25):
        strategy["calorie_approach"] = "moderate calorie deficit"
        strategy["protein"] = "high"
        strategy["fiber"] = "high"
        strategy["focus"] += [
            "vegetables",
            "lean proteins",
            "whole foods"
        ]

    # Muscle gain
    if "muscle_gain" in goals:
        strategy["calorie_approach"] = "small calorie surplus"
        strategy["protein"] = "very high"
        strategy["focus"] += [
            "protein-rich foods",
            "complex carbohydrates"
        ]


    # Bad food habits from check-ins
    if "heavy_meals" in food_patterns:
        strategy["avoid"].append("fried and very fatty foods")


    if "high_sugar" in food_patterns:
        strategy["avoid"].append("sugary drinks and sweets")


    # Digestion
    if "digestion" in health:
        strategy["focus"] += [
            "fiber",
            "water",
            "probiotic foods"
        ]
        strategy["avoid"].append("ultra-processed foods")
    
    if (
        "constipation" in health
        or "fiber_support" in health
    ):
        strategy["fiber"] = "high"
        strategy["focus"] += [
            "gradually increased fiber",
            "adequate water",
            "vegetables",
            "whole grains",
        ]

    if "appetite_control" in health:
        strategy["focus"] += [
            "high-satiety meals",
            "protein-rich foods",
            "fiber-rich foods",
        ]

    if (
        "glycemic_balance" in health
        or "insulin_sensitivity" in health
        or "sugar_cravings" in health
    ):
        strategy["avoid"] += [
            "sugary drinks",
            "large sweet portions",
            "refined carbohydrates",
        ]
        strategy["focus"] += [
            "non-starchy vegetables",
            "lean proteins",
            "high-fiber carbohydrates",
            "balanced meal timing",
        ]
    # Medical conditions




    strategy["avoid"] = list(dict.fromkeys(strategy["avoid"]))
    strategy["focus"] = list(dict.fromkeys(strategy["focus"]))
    # Age-based nutrition
    if age and age >= 60:
        strategy["protein"] = "high"
        strategy["focus"] += [
            "adequate protein",
            "calcium-rich foods",
            "vitamin D sources",
            "hydration"
        ]

    # Sex-based nutrition
    if sex == "female":
        strategy["focus"] += [
            "iron-rich foods",
            "calcium-rich foods"
        ]

    if sex == "male":
        strategy["focus"] += [
            "adequate portions based on activity level"
        ]

# BMI-based nutrition
    if bmi and bmi < 18.5:
        strategy["calorie_approach"] = "healthy calorie surplus"
        strategy["protein"] = "high"
        strategy["focus"] += [
            "energy-dense healthy foods",
            "protein-rich meals",
            "complex carbohydrates"
        ]

    elif bmi and 18.5 <= bmi < 25:
        strategy["calorie_approach"] = "maintenance"
        strategy["focus"] += [
            "balanced meals",
            "vegetables",
            "whole grains"
        ]

    elif bmi and bmi >= 30:
        strategy["calorie_approach"] = "gradual calorie deficit"
        strategy["focus"] += [
            "high-satiety meals",
            "lean proteins",
            "fiber-rich foods"
        ]
    
    
    # Activity/steps influence nutrition
    if (
        avg_steps is not None
        and avg_steps < 5000
        and ("weight_loss" in goals or (bmi and bmi >= 25))
    ):
        strategy["focus"] += [
            "high-satiety meals",
            "lean proteins",
            "vegetables"
        ]
        strategy["avoid"] += [
            "large portions of bread and fried foods"
        ]

    if (
        (avg_steps is not None and avg_steps >= 7500)
        or activity_level == "high"
    ):
        strategy["focus"] += [
            "recovery meals",
            "adequate carbohydrates",
            "hydration"
        ]

    if (
        "muscle_gain" in goals
        and (
            (avg_steps is not None and avg_steps >= 5000)
            or activity_level in ["moderate", "high"]
        )
    ):
        strategy["focus"] += [
            "post-workout protein",
            "complex carbohydrates after training"
        ]

    # Mood influence nutrition
    moods_text = normalize_moods(recent_moods)
    if "low_energy" in moods_text:
        strategy["focus"] += [
            "iron-rich foods",
            "regular meal timing",
            "adequate hydration"
        ]

    if "stress" in moods_text or "anxious" in moods_text or "قلق" in moods_text:
        strategy["focus"] += [
            "regular meals",
            "magnesium-rich foods",
            "omega-3 sources"
        ]

    strategy["avoid"] = list(dict.fromkeys(strategy["avoid"]))
    strategy["focus"] = list(dict.fromkeys(strategy["focus"]))
    
    # Final nutrition safety override
    if "cholesterol" in conditions_text:
        strategy["avoid"] += [
            "fried foods",
            "high saturated fat foods"
        ]
        strategy["focus"] += [
            "fiber",
            "lean proteins",
            "healthy fats"
        ]


    if "pregnancy" in conditions_text or "breastfeeding" in conditions_text:
        strategy["calorie_approach"] = "doctor-approved balanced nutrition"
        strategy["avoid"] += ["supplements without medical advice", "extreme dieting"]
        strategy["focus"] += ["balanced meals", "hydration", "doctor-approved nutrition"]

    if "diabetes" in conditions_text:
        strategy["avoid"] += ["sugary drinks", "large sweet portions"]
        strategy["focus"] += ["non-starchy vegetables", "lean protein", "quality carbohydrates"]

    if "hypertension" in conditions_text or "high blood pressure" in conditions_text:
        strategy["avoid"] += ["high sodium foods", "processed foods"]
        strategy["focus"] += ["DASH-style meals", "vegetables", "fruits", "whole grains"]

    strategy["avoid"] = list(dict.fromkeys(strategy["avoid"]))
    strategy["focus"] = list(dict.fromkeys(strategy["focus"]))

    return strategy
def find_best_bundle_for_recommendations(
    recommended_products: list[dict],
    bundle_offers_db: list,
) -> dict | None:
    recommended_names = {
        item.get("product")
        for item in recommended_products
        if item.get("product")
    }

    matching_bundles = []

    for bundle in bundle_offers_db:
        if not bundle.get("active", True):
            continue

        bundle_products = set(
            bundle.get("products") or []
        )

        overlap = recommended_names.intersection(
            bundle_products
        )

        if len(overlap) >= 2:
            matching_bundles.append({
                **bundle,
                "_overlap_count": len(overlap),
            })

    if not matching_bundles:
        return None

    matching_bundles.sort(
        key=lambda bundle: (
            bundle.get("_overlap_count", 0),
            (
                bundle.get("old_price", 0)
                - bundle.get("new_price", 0)
            ),
        ),
        reverse=True,
    )

    best = matching_bundles[0]
    best.pop("_overlap_count", None)

    return best
def rerank_products_with_llm(
    ranked_products: list[dict],
    signals: dict,
    merged_profile: dict,
) -> list[dict]:
    if client is None or not ranked_products:
        return ranked_products

    candidates = []

    for item in ranked_products[:5]:
        product = item["product"]

        candidates.append({
            "name": item["name"],
            "score": item["score"],
            "matched_signals": item["matched_signals"],
            "tags": list(
                normalize_product_tags(product)
            ),
            "benefits": (
                product.get("benefits")
                or []
            )[:3],
            "precautions": (
                product.get("precautions")
                or []
            )[:3],
        })

    prompt = f"""
You are reranking already eligible Shifa products.

Do not add products.
Do not remove medical precautions.
Do not recommend products as treatments or cures.

Return ONLY JSON:
{{
  "ordered_product_names": []
}}

User profile:
{json.dumps(merged_profile, ensure_ascii=False)}

Detected signals:
{json.dumps(signals, ensure_ascii=False)}

Eligible candidates:
{json.dumps(candidates, ensure_ascii=False)}

Rank the candidates from best to least relevant.
Only use product names present in eligible candidates.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. "
                        "Never invent a product."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        data = json.loads(
            response.choices[0].message.content.strip()
        )

        ordered_names = data.get(
            "ordered_product_names",
            [],
        )

        by_name = {
            item["name"]: item
            for item in ranked_products
        }

        ordered = [
            by_name[name]
            for name in ordered_names
            if name in by_name
        ]

        remaining = [
            item
            for item in ranked_products
            if item["name"] not in ordered_names
        ]

        return ordered + remaining

    except Exception as exc:
        print("Product reranking failed:", exc)
        return ranked_products
def apply_digestive_product_priority(
    ranked_products: list[dict],
    signals: dict,
) -> list[dict]:
    health = set(signals.get("health_interests", []) or [])

    psyllium_priority = bool(
        health.intersection({
            "constipation",
            "fiber_support",
            "appetite_control",
        })
    )

    colon_priority = bool(
        health.intersection({
            "digestion",
            "bloating_gas",
        })
    )

    if psyllium_priority and not colon_priority:
        ranked_products = sorted(
            ranked_products,
            key=lambda item: (
                item["name"] != "Psyllium",
                -item["score"],
            ),
        )

    elif colon_priority and not psyllium_priority:
        ranked_products = sorted(
            ranked_products,
            key=lambda item: (
                item["name"] != "Colon Detox",
                -item["score"],
            ),
        )

    return ranked_products

def apply_product_goal_priority(
    ranked_products: list[dict],
    signals: dict,
    merged_profile: dict,
    bmi: float | None,
) -> list[dict]:
    """
    Apply deterministic business priorities after scoring.

    Rules:
    - General weight-loss goal -> Slim Pack
    - Glycemic/insulin/sugar signals -> Berberine
    - Berberine must not replace Slim Pack for weight loss alone
    """

    health = {
        str(item).lower().strip()
        for item in (
            signals.get("health_interests", [])
            or []
        )
    }

    recommended_signals = {
        str(item).lower().strip()
        for item in (
            signals.get(
                "recommended_product_signals",
                [],
            )
            or []
        )
    }

    goals = merged_profile.get("goals") or []

    if not isinstance(goals, list):
        goals = [goals]

    goals = {
        str(goal).lower().strip()
        for goal in goals
        if str(goal).strip()
    }

    all_signals = (
        health
        | recommended_signals
        | goals
    )

    glycemic_signals = {
        "glycemic_balance",
        "insulin_sensitivity",
        "sugar_cravings",
        "blood sugar",
        "glucose",
        "insulin",
        "diabetes",
    }

    weight_signals = {
        "weight_loss",
        "weight loss",
        "weight_management",
        "weight management",
        "slimming",
    }

    has_glycemic_need = bool(
        all_signals.intersection(
            glycemic_signals
        )
    )

    has_weight_goal = (
        bool(
            all_signals.intersection(
                weight_signals
            )
        )
        or (
            bmi is not None
            and bmi >= 25
        )
    )

    # Explicit glycemic need:
    # Berberine can be prioritized.
    if has_glycemic_need:
        return sorted(
            ranked_products,
            key=lambda item: (
                item["name"]
                != "Berberine & Ceylon Cinnamon",
                -item["score"],
            ),
        )

    # General weight-loss need without
    # glycemic signals:
    # Slim Pack must be prioritized.
    if has_weight_goal:
        return sorted(
            ranked_products,
            key=lambda item: (
                item["name"] != "Slim Pack",
                -item["score"],
            ),
        )

    return ranked_products

def build_product_reason(
    product_name: str,
    product: dict,
    matched_signals: list[str],
    language: str,
) -> str:
    fixed_reason_key = {
        "Colon Detox": "colon_reason",
        "Liver Detox": "liver_reason",
        "Slim Pack": "slim_reason",
        "Blood Detox": "blood_reason",
        "Lung Detox": "lung_reason",
        "Berberine & Ceylon Cinnamon": "berberine_reason",
        "Psyllium": "psyllium_reason",
    }.get(product_name)

    if not fixed_reason_key:
        return ""

    return translated_text(
        language,
        fixed_reason_key,
    )
def score_products(
    product_db: dict,
    signals: dict,
    merged_profile: dict,
    bmi: float | None,
) -> list[dict]:
    user_signals = build_user_product_signals(
        signals=signals,
        merged_profile=merged_profile,
        bmi=bmi,
    )

    direct_product_signals = {
        str(item).lower().strip()
        for item in (
            signals.get("recommended_product_signals", [])
            or []
        )
        if str(item).strip()
    }

    scored_products = []

    for product_name, product in product_db.items():
        product_tags = normalize_product_tags(product)

        aliases = {
            alias.lower().strip()
            for alias in PRODUCT_SIGNAL_ALIASES.get(
                product_name,
                set(),
            )
        }

        searchable_terms = product_tags | aliases

        matches = sorted({
            signal
            for signal in user_signals
            if any(
                signal == term
                or signal in term
                or term in signal
                for term in searchable_terms
            )
        })

        score = len(matches) * 3

        health = {
            str(item).lower().strip()
            for item in (
                signals.get("health_interests", [])
                or []
            )
        }

        goals = merged_profile.get("goals") or []

        if not isinstance(goals, list):
            goals = [goals]

        goals = {
            str(goal).lower().strip()
            for goal in goals
            if str(goal).strip()
        }

        glycemic_signals = {
            "glycemic_balance",
            "insulin_sensitivity",
            "sugar_cravings",
            "blood sugar",
            "glucose",
            "insulin",
            "diabetes",
        }

        weight_signals = {
            "weight_loss",
            "weight loss",
            "weight_management",
            "weight management",
            "slimming",
        }

        has_glycemic_need = bool(
            health.intersection(glycemic_signals)
        )

        has_weight_goal = (
            bool(goals.intersection(weight_signals))
            or (
                bmi is not None
                and bmi >= 25
            )
        )

        if (
            product_name == "Slim Pack"
            and has_weight_goal
        ):
            score += 8

        if (
            product_name
            == "Berberine & Ceylon Cinnamon"
        ):
            if has_glycemic_need:
                score += 8

            elif has_weight_goal:
        # Weight loss alone is not enough
        # to prioritize Berberine.
                score -= 4




        # Psyllium can support a weight-loss plan through fiber and satiety.
        if (
            product_name == "Psyllium"
            and "weight_loss" in user_signals
        ):
            score += 3

            if "satiety support for weight loss" not in matches:
                matches.append(
                    "satiety support for weight loss"
                )
        normalized_product_name = product_name.lower().strip()

        if normalized_product_name in direct_product_signals:
            score += 6

            if normalized_product_name not in matches:
                matches.append(normalized_product_name)

        if score > 0:
            scored_products.append({
                "name": product_name,
                "product": product,
                "score": score,
                "matched_signals": matches,
            })

    return sorted(
        scored_products,
        key=lambda item: item["score"],
        reverse=True,
    )
def build_exercise_strategy(merged_profile, signals):

    strategy = {
        "main_activity": "general movement",
        "duration": "30 minutes",
        "frequency": "3 times per week",
        "intensity": "moderate",
        "exercise_type": "general",
        "focus": "",
        "step_goal": "",
        "today_program": {},
        "body_focus": [],
        "exercise_examples": [],
        "limitations": [],
        "reasoning": []
    }

    avg_steps = merged_profile.get("average_steps_last_7_days")
    activity = signals.get("activity_level")
    age = merged_profile.get("age")
    goals = merged_profile.get("goals", [])
    conditions = merged_profile.get("medical_conditions", [])
    conditions_text = " ".join(conditions).lower() if isinstance(conditions, list) else str(conditions).lower()
    bmi = merged_profile.get("bmi")
    sex = merged_profile.get("sex")
    food_patterns = signals.get("food_patterns", [])
    health = signals.get("health_interests", [])
   
    recent_moods = merged_profile.get("recent_moods", [])
    consistency = merged_profile.get("consistency_score")
    recurring_activity_patterns = merged_profile.get("recurring_activity_patterns", [])
    low_activity_count = recurring_activity_patterns.count("low")
    moderate_activity_count = recurring_activity_patterns.count("moderate")
    high_activity_count = recurring_activity_patterns.count("high")
    
    if sex == "female":
        strategy["reasoning"].append(
            "Program adapted to female physiological characteristics and personal goals."
        )

    elif sex == "male":
        strategy["reasoning"].append(
            "Program adapted to male physiological characteristics and personal goals."
        )

    
    moods_text = normalize_moods(recent_moods)

# Step-based base level

    if avg_steps is None:
        strategy["focus"] = "activity level not yet measured"
        strategy["step_goal"] = ""
        strategy["reasoning"].append(
            "No step history available: avoid assuming sedentary behavior"
        )

    
    elif avg_steps < 5000:
        strategy["focus"] = "increase daily movement"
        strategy["step_goal"] = "start with 5000–7000 steps/day"
        strategy["duration"] = "20-30 minutes"
        strategy["intensity"] = "light"
    elif avg_steps < 7500:
        strategy["focus"] = "progressive improvement"
        strategy["step_goal"] = "reach 7000–10000 steps/day"
        strategy["intensity"] = "light to moderate"
    elif avg_steps < 10000:
        strategy["focus"] = "maintain good activity level"
        strategy["step_goal"] = "maintain 8000–10000 steps/day"
    else:
        strategy["focus"] = "maintain excellent activity level"
        strategy["step_goal"] = "maintain current activity"

# Activity-level signal from check-ins / LLM signals
    if activity == "low":
        strategy["intensity"] = "light"
        strategy["duration"] = "20-30 minutes"
        strategy["reasoning"].append("Low activity level: start progressively")

    elif activity == "moderate":
        strategy["intensity"] = "light to moderate"
        strategy["reasoning"].append("Moderate activity level: maintain and progress gradually")

    elif activity == "high":
        strategy["intensity"] = "moderate to high"
        strategy["reasoning"].append("High activity level: allow more advanced session if safe")
    
    # Long-term activity behavior from memory
    if low_activity_count >= 2:
        strategy["duration"] = "10-20 minutes"
        strategy["intensity"] = "light"
        strategy["exercise_examples"].append("short walking breaks")
        strategy["reasoning"].append("Recurring low activity pattern: build habit gradually")

    elif high_activity_count >= 2:
        strategy["reasoning"].append("Recurring high activity pattern: allow progression and recovery balance")

    elif moderate_activity_count >= 2:
        strategy["reasoning"].append("Recurring moderate activity pattern: maintain consistency and progress slowly")
# Goal-based exercise type
    if "muscle_gain" in goals:
        strategy["exercise_type"] = "strength training"
        strategy["main_activity"] = "resistance training"
        strategy["duration"] = "40-60 minutes"
    elif "weight_loss" in goals or (bmi and bmi >= 25):
        strategy["exercise_type"] = "cardio + light strength"
        strategy["main_activity"] = "cardio and full-body movement"
    else: 
        strategy["exercise_type"] = "wellness movement"
        strategy["main_activity"] = "walking and mobility"
    # BMI adaptation
    if bmi and bmi >= 30:
        strategy["intensity"] = "light to moderate"
        strategy["duration"] = "20-30 minutes"
        strategy["exercise_examples"].append("low-impact cardio")
        strategy["reasoning"].append("High BMI: prefer low-impact progressive movement")

    elif bmi and bmi < 18.5:
        strategy["exercise_type"] = "strength + mobility"
        strategy["exercise_examples"].append("light resistance training")
        strategy["reasoning"].append("Low BMI: avoid excessive cardio and support strength")
# Health / behavior adaptation
    if "heavy_meals" in food_patterns:
        strategy["exercise_examples"].append("walking after meals")
        strategy["reasoning"].append("Heavy meal pattern detected")

    if "high_sugar" in food_patterns:
        strategy["exercise_examples"].append("light cardio")
        strategy["reasoning"].append("High sugar pattern detected")

    if "digestion" in health:
        strategy["exercise_examples"].append("gentle walking after meals")
        strategy["reasoning"].append("Digestive concern detected")

    if "stress_anxiety" in health or "sleep" in health or "low_mood" in moods_text:
        strategy["exercise_examples"] += ["pilates", "stretching", "breathing exercises", "relaxing walk"]
        strategy["intensity"] = "light to moderate"
        strategy["reasoning"].append("Mood/stress/sleep adaptation")

    if "energetic" in moods_text and not ("pregnancy" in conditions_text or "breastfeeding" in conditions_text):
        strategy["reasoning"].append("Good energy detected: allow slightly more progressive session")

# Consistency adaptation
    if consistency is not None and consistency < 50:
        strategy["duration"] = "10-20 minutes"
        strategy["intensity"] = "light"
        strategy["reasoning"].append("Low consistency: start with simple achievable activity")
    elif consistency is not None and consistency >= 75:
        strategy["reasoning"].append("High consistency: user can progress gradually")

# Daily body/type rotation
    body_rotation = ["cardio", "lower body", "upper body + core", "mobility", "full body"]
    rotation_index = date.today().weekday() % len(body_rotation)
    rotated_focus = body_rotation[rotation_index]

# Today's program selector
    if "stress_anxiety" in health or "sleep" in health or "low_mood" in moods_text:
        today_focus = "stress relief / mobility"
        today_activity = "pilates, stretching, breathing exercises, or relaxing walk"
    elif "muscle_gain" in goals:
        today_focus = rotated_focus if rotated_focus != "cardio" else "full body"
        today_activity = f"{today_focus} strength exercises adapted to your level"
    elif "weight_loss" in goals or (bmi and bmi >= 25):
        today_focus = rotated_focus
        today_activity = f"{today_focus} session with cardio and light strength"
    else:
        today_focus = rotated_focus
        today_activity = f"{today_focus} wellness movement"

    strategy["today_program"] = {
        "focus": today_focus,
        "activity": today_activity,
        "duration": strategy.get("duration"),
        "intensity": strategy.get("intensity"),
        "step_goal": strategy.get("step_goal"),
        "extra": list(dict.fromkeys(strategy.get("exercise_examples", [])))[:3],
        
 
        "frequency": strategy.get("frequency"),
    
    }



    # Final safety override: medical conditions have highest priority
    if "pregnancy" in conditions_text or "breastfeeding" in conditions_text:
        strategy["main_activity"] = "doctor-approved gentle movement"
        strategy["duration"] = "15-30 minutes"
        strategy["frequency"] = "as approved by doctor"
        strategy["intensity"] = "light"
        strategy["step_goal"] = "gentle daily movement only if approved"
        strategy["exercise_examples"] = ["doctor-approved walking", "gentle mobility"]
        strategy["reasoning"].append("Pregnancy/breastfeeding safety override")
        strategy["today_program"] = {
            "focus": "safety and gentle movement",
            "activity": "doctor-approved walking or gentle mobility",
            "duration": "15-30 minutes",
            "intensity": "light",
            "step_goal": "gentle movement only if approved",
            "extra": ["avoid intense exercise", "rest if tired"],
            "frequency": strategy["frequency"]
        }



    elif "hypertension" in conditions_text or "high blood pressure" in conditions_text:
        strategy["main_activity"] = "walking and low-impact cardio"
        strategy["intensity"] = "light to moderate"
        strategy["reasoning"].append("Hypertension safety adaptation")
        strategy["today_program"] = {
            "focus": "low-impact cardio",
            "activity": "walking or low-impact cardio",
            "duration": strategy.get("duration"),
            "intensity": "light to moderate",
            "step_goal": strategy.get("step_goal"),
            "extra": ["avoid sudden intense effort"],
            "frequency": strategy.get("frequency")

        }

    elif age and age >= 60:
        strategy["main_activity"] = (
            "walking, mobility, balance, and light resistance training"
        )
        strategy["duration"] = "20-30 minutes"
        strategy["intensity"] = "light to moderate"
        strategy["frequency"] = "most days of the week"
        strategy["reasoning"].append(
            "Older adult safety adaptation"
        )

        strategy["today_program"] = {
            "focus": "balance, mobility, and gentle strength",
            "activity": (
                "walking, balance exercises, mobility, "
                "and light resistance training"
            ),
            "duration": strategy["duration"],
            "frequency": strategy["frequency"],
            "intensity": strategy["intensity"],
            "step_goal": strategy.get("step_goal"),
            "extra": [
                "use support if balance is uncertain",
                "avoid sudden high-impact effort",
                "stop if pain or dizziness occurs",
            ],
        }
    return strategy
def build_daily_variation_context() -> dict:
    today = date.today()

    meal_themes = [
        "fresh salad and grilled protein",
        "healthy Tunisian soup and protein",
        "balanced rice or whole-grain meal",
        "egg-based meal with vegetables",
        "legume-based meal",
        "grilled fish and vegetables",
        "healthy Tunisian traditional meal",
    ]

    exercise_themes = [
        "cardio",
        "lower body",
        "upper body and core",
        "mobility and recovery",
        "full body",
        "walking and light strength",
        "stretching and active recovery",
    ]

    index = today.toordinal()

    return {
        "date": str(today),
        "meal_theme": (
            meal_themes[
                index % len(meal_themes)
            ]
        ),
        "exercise_theme": (
            exercise_themes[
                index % len(exercise_themes)
            ]
        ),
        "variation_number": index,
    }
def generate_dynamic_plan_with_llm(
    merged_profile: dict,
    signals: dict,
    decision: dict,
    nutrition_strategy: dict,
    exercise_strategy: dict
    
):
    fallback = {
        "meal_recommendations": [],
        "exercise_recommendations": [],
        "daily_actions": [],
        "warnings": []
    }
    daily_variation = (
        build_daily_variation_context()
    )
    language = normalize_language(
        merged_profile.get("language", "ar")
    )

    language_name = LANGUAGE_NAMES[language]

    language_instruction = f"""
    STRICT OUTPUT LANGUAGE: {language_name}

    Every user-visible JSON value must be written only in {language_name}:
    - behavioral_insight
    - meal titles
    - meal recommendations
    - meal reasons
    - exercise titles
    - exercise recommendations
    - exercise reasons
    - daily actions
    - daily-action reasons
    - motivation_message

    Do not mix Arabic, French, and English.
    Product names such as Slim Pack, Colon Detox, Liver Detox, Blood Detox,
    Lung Detox, Psyllium, and Berberine & Ceylon Cinnamon may remain unchanged.
    """
    

    if client is None:
        return fallback

    prompt = f"""
You are a wellness recommendation assistant for Shifa.
{language_instruction}


Today's daily variation context:
{json.dumps(
    daily_variation,
    ensure_ascii=False,
)}

Daily variation rules:
- Keep the same health, nutrition and exercise strategy.
- Change the concrete meal examples according to today's meal theme.
- Change the concrete workout examples according to today's exercise theme.
- Do not change medical safety rules.
- Do not change the main user goal.
- Do not change product recommendations only for variety.
- Recommendations generated on the same date should follow the same daily theme.
- Do not generate exactly the same meal titles on consecutive days unless medically necessary.
- Do not generate exactly the same exercise session on consecutive days unless medically necessary.
- Avoid repeating the exact same meal titles and workout title every day.

Generate personalized:
1. meal recommendations
2. exercise recommendations
3. daily wellness actions

The recommendations MUST consider ALL available information:

1. User profile:
- age
- sex
- weight
- height
- BMI
- goals
- medical conditions

2. Recent 7-day behavior:
- detected food patterns from recent check-ins
- detected activity level
- consistency and behavioral trends
- mood patterns
- average daily steps

3. Long-term personalization:
- last 90 days useful chatbot interactions
- user memory and recurring patterns

4. Rule-based strategies:
- nutrition_strategy generated from scientific rules
- exercise_strategy generated from scientific rules

Do not ignore any important factor.

User profile:
{json.dumps(merged_profile, ensure_ascii=False)}

Extracted signals:
{json.dumps(signals, ensure_ascii=False)}

Decision layer:
{json.dumps(decision, ensure_ascii=False)}
Nutrition strategy:
{json.dumps(nutrition_strategy, ensure_ascii=False)}

Exercise strategy:
{json.dumps(exercise_strategy, ensure_ascii=False)}

Rules:
Output language: {language}

Language rules:
- If language = "fr", write everything in French.
- If language = "ar", write everything in Arabic or Tunisian Arabic.
- Do not mix French and Arabic in the same recommendation.
- Be practical and personalized.
- Avoid generic robotic recommendations.
- Adapt meals to detected food patterns.
- Adapt exercise intensity to activity level.
- Keep recommendations realistic.
Advanced reasoning rules:
- Use recent_chat_topics from the last 90 days only as long-term personalization signals.
- Do not repeat or summarize the whole chatbot conversation.
- Use chat history only to detect recurring concerns, preferences, products discussed, and repeated difficulties.
- If recent check-ins show improvement after previous poor habits, encourage progress and suggest maintaining the new habit.
- If behavior is consistently good, suggest maintenance and small optimizations.
- If behavior is worsening, give more corrective but supportive recommendations.
- Do not recommend extreme diets, unrealistic exercises, or medical claims. Prefer realistic habits adapted to the user profile and recent behavior.
- Use age to adapt exercise intensity and recovery advice.
- The exercise title and description must accurately match the main workout.
- If the workout includes strength exercises, do not describe it as only stretching or recovery.
- Younger users can tolerate more progressive activity suggestions.
- Older users should receive more gradual and recovery-focused advice.
- Use food_patterns, activity_level, average steps, mood, consistency, health interests, and today_program to personalize recommendations.
- Use BMI to adapt calorie-balance recommendations.
- High BMI -> focus on sustainable habits and progressive movement.
- Low BMI -> avoid aggressive calorie restriction.
- If medical_conditions exist, adapt meals and exercise safely.
- Do not give diagnosis or treatment.
- For diabetes: avoid high-sugar recommendations.
- For hypertension: avoid high-salt recommendations.
- For pregnancy, breastfeeding, chronic disease, or medication use: recommend medical advice before supplements.
- Do not always choose the same body focus. Use exercise_strategy.today_program.focus as today’s target and vary body focus across days when medically safe.
- Generate ONLY today’s exercise program, not a weekly plan.
- Use exercise_strategy.today_program as the main exercise recommendation.
- Today’s program must vary the exercise type/body focus depending on all available data: goal, BMI, age, sex, medical conditions, moods, average steps, activity_level, food patterns, health interests, recurring behaviors, chat history, and consistency.
- Mention body focus when available: upper body, lower body, core, cardio, mobility, recovery.
- Respect intensity, duration, frequency, step_goal, limitations and medical safety rules.
Generate ONE structured workout for today.

Return:

- title
- description
- duration
- frequency
- intensity
- warmup
- 3 to 5 main exercises
- cooldown
- reason

Example:

{{
"title":"Today's Walking Session",
"description":"A beginner-friendly cardio session.",
"duration":"30 minutes",
"frequency":"Today",
"intensity":"Light",
"warmup":"5 minutes easy walking",
"main_workout":[
"15 min brisk walking",
"10 squats",
"10 wall push-ups"
],
"cooldown":"5 minutes stretching",
"reason":"Suitable for your current activity level."
}}

Use exercise_strategy.today_program as the main guidance.

The session must include:
- A warm-up adapted to the user's condition (3–10 minutes)
- 3–5 exercises with precise sets/repetitions or duration
- A cool-down with stretching, breathing, or mobility exercises

Adapt the exercises according to:
- Age
- Sex (only for physiological context, never limit body parts based on sex)
- BMI and current fitness level
- Goal (weight loss, muscle gain, general wellness)
- Medical conditions and safety limitations (highest priority)
- Average steps and activity level from the last 7 days
- Mood and consistency level
- Health interests and long-term behavior

If the user has low energy, low consistency, or low activity, propose simpler exercises.

If the user is energetic and has no medical restrictions, propose more challenging exercises.
The deterministic rule engine has already selected today's focus, duration,
intensity, step goal, medical limitations, and suitable activity category.

You must generate the exact warm-up, exercises, sets/repetitions, and cooldown
within these constraints.

Never change:
- exercise_strategy.today_program.focus
- exercise_strategy.today_program.duration
- exercise_strategy.today_program.intensity
- exercise_strategy.today_program.step_goal
- medical safety overrides

Do not generate an exercise that conflicts with the user's medical conditions,
age, BMI, activity level, consistency, or today's selected focus.
Do not recommend unsafe exercises. Medical conditions always override fitness goals.
Cultural adaptation rules:
- Adapt meal recommendations to Tunisian food culture when appropriate.
- Prefer realistic Tunisian healthy meals instead of only international meals.
- Keep the recommendations compatible with the nutrition_strategy.
- You may suggest traditional Tunisian dishes with healthier adaptations.

Examples:
- Couscous with vegetables and lean chicken instead of high-fat couscous.
- Lablabi with moderate bread portions and good protein sources.
- Ojja with vegetables and controlled oil.
- Slata mechouia with tuna or eggs.
- Grilled fish with vegetables.
- Chorba with balanced portions.
- Frik soup.
- Healthy Tunisian salads.
- Fruits, nuts, yogurt, and dates can be suggested only when compatible with medical conditions. For diabetes, prefer unsweetened yogurt, nuts, and low-glycemic fruits, and avoid adding honey or excessive dates.

For weight loss:
- Reduce portions of bread, sweets, fried foods, and sugary drinks.
- Prefer grilled, steamed, and home-cooked meals.

For muscle gain:
- Increase protein sources such as eggs, chicken, fish, dairy, legumes, and Tunisian dishes rich in protein.

For diabetes or blood sugar concerns:
- Prefer low glycemic and high-fiber choices.
- Control bread, pastries, and sugary dessert portions.

Do not criticize traditional Tunisian food. Adapt it to make it healthier.

IMPORTANT:
Do not decide nutrition or exercise strategy by yourself.

The nutrition_strategy and exercise_strategy are generated by a deterministic rule engine based on BMI, goals, age, activity level, health concerns and recent behavior.

Your role is only to transform these strategies into:
- realistic meals
- suitable exercises
- practical daily actions
The behavioral_insight must speak directly to the user.
Do not say "the user".
Use a friendly coaching tone.
For French, address the person with "vous".
For Arabic, speak directly and naturally to the person.
For English, use "you".
Keep it to 1–3 short sentences.
Always follow the provided strategies.
Do not recommend extreme diets or unrealistic workouts.

- Use consistency_score:
  - low consistency -> suggest small achievable actions.
  - medium consistency -> encourage stabilization.
  - high consistency -> suggest progression and stronger discipline.
  - If consistency is low:
      be supportive and avoid harsh recommendations.

  - If consistency is high:
      use more motivating/progressive tone.
- Detect repeated unhealthy habits from recurring_food_patterns.
- Detect sedentary lifestyle from recurring_activity_patterns.

Do NOT generate supplement safety warnings.
Do NOT generate medical-condition warnings.
These warnings are already handled by the rule engine.
- If user repeatedly reports heavy meals or high sugar:
  recommendations should directly adapt to these habits.

- Avoid repeating identical recommendations every day.
- Recommendations should feel contextual, progressive, human, and realistic.
- Return ONLY valid JSON.

JSON schema:
{{
  "behavioral_insight": "",
  "meal_recommendations": [
    {{
      "title": "",
      "recommendation": "",
      "reason": ""
    }}
  ],
  
  "exercise_recommendations":[
    {{
      "title":"",
      "description":"",
      "duration":"",
      "frequency":"",
      "intensity":"",
      "warmup":"",
      "main_workout":[
        "",
        "",
        ""
      ],
      "cooldown":"",
      "reason":""
    }}
  ],

  "daily_actions": [
    {{
      "action": "",
      "reason": ""
    }}
  ],
  "motivation_message": "",
  "warnings": []
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        return json.loads(content)

    except Exception:
        return fallback
    
def build_recommendation_decision(
    signals: dict,
    merged_profile: dict,
    bmi: float | None,
    quantity_offers_db: dict,
    bundle_offers_db: list
) -> dict:
    health_interests = signals.get("health_interests", [])
    activity_level = signals.get("activity_level", "unknown")
    food_patterns = signals.get("food_patterns", [])
    priority_need = signals.get("detected_priority_need")
    goals = merged_profile.get("goals", [])
    goals_text = " ".join(goals).lower() if isinstance(goals, list) else str(goals).lower() 
    age = merged_profile.get("age")
    consistency = merged_profile.get("consistency_score")
    conditions = merged_profile.get("medical_conditions", [])
    language = normalize_language(
        merged_profile.get("language", "ar")
    )
    decision = {
        "priority_focus": None,
        "recommended_products": [],
        "meal_plan": {
            "breakfast": None,
            "lunch": None,
            "dinner": None,
            "snack": None,
        },
        "exercise_plan": {
            "main_activity": None,
            "duration": None,
            "intensity": None,
            "frequency": None,
        },
        "daily_actions": [],
        "warnings": [],
        "upsell_strategy": None,
        "confidence_score": 0,
        "used_signals": {
            "memory": False,
            "profile": False,
            "daily_checkin": False,
            "bmi": False,
            "activity": False,
        }
    }

    decision["used_signals"]["memory"] = bool(
        merged_profile.get("past_recommended_products")
        or merged_profile.get("recurring_food_patterns")
        or merged_profile.get("recurring_activity_patterns")
        or merged_profile.get("last_detected_issue")
    )
    if goals:
        decision["used_signals"]["profile"] = True

    if bmi:
        decision["used_signals"]["bmi"] = True

    if activity_level != "unknown":
        decision["used_signals"]["activity"] = True


    # -------------------------
    # 1. Decide priority focus
    # -------------------------
    if health_interests:
        decision["priority_focus"] = health_interests[0]

        decision["used_signals"]["daily_checkin"] = bool(
            merged_profile.get("recent_checkins")
        )

    elif priority_need:
        decision["priority_focus"] = priority_need

        decision["used_signals"]["daily_checkin"] = bool(
            merged_profile.get("recent_checkins")
        )

    elif (
        "weight_loss" in goals_text
        or "weight loss" in goals_text
        or "weightloss" in goals_text
    ):
        decision["priority_focus"] = "weight_loss"
        decision["used_signals"]["profile"] = True

    elif "muscle_gain" in goals_text:
        decision["priority_focus"] = "muscle_gain"
        decision["used_signals"]["profile"] = True

    elif "general_wellness" in goals_text:
        decision["priority_focus"] = "general_wellness"

    elif bmi and bmi >= 25:
        decision["priority_focus"] = "weight_management"
        decision["used_signals"]["bmi"] = True

    else:
        decision["priority_focus"] = "general_wellness"
# Advanced behavioral logic
# -------------------------

    if (
        age
        and age >= 50
        and activity_level == "low"
    ):
        decision["daily_actions"].append({
            "action": translated_text(language, "movement_action"),
            "reason": translated_text(language, "movement_reason"),
        })
    if (
        bmi and bmi >= 28
        and "high_sugar" in food_patterns
        and consistency < 50
    ):
        decision["priority_focus"] = "weight_management"

    if "stress_anxiety" in health_interests:
        decision["daily_actions"].append({
            "action": translated_text(language, "stress_action"),
            "reason": translated_text(language, "stress_reason"),
        })
    if consistency is not None and consistency < 40:
        decision["daily_actions"].append({
            "action": translated_text(language, "small_habits_action"),
            "reason": translated_text(language, "small_habits_reason"),
        })
    conditions_text = " ".join(conditions).lower() if isinstance(conditions, list) else str(conditions).lower()
    
    if "pregnancy" in conditions_text or "breastfeeding" in conditions_text:
        decision["priority_focus"] = "pregnancy_safe_wellness"
        decision["warnings"].append(
            translated_text(language, "pregnancy_warning")
        )
        decision["confidence_score"] = 70
        return decision
    # -------------------------
    # 2. Product decision
    # -------------------------
    language = normalize_language(
        merged_profile.get("language", "ar")
    )

    product_db = get_product_knowledge_dict()

    ranked_products = score_products(
        product_db=product_db,
        signals=signals,
        merged_profile=merged_profile,
        bmi=bmi,
    )

    ranked_products = apply_digestive_product_priority(
        ranked_products,
        signals,
    )

    ranked_products = apply_product_goal_priority(
        ranked_products=ranked_products,
        signals=signals,
        merged_profile=merged_profile,
        bmi=bmi,
    )

    # Optional AI reranking:
    # It can only reorder products already approved by the deterministic scorer.
    if ranked_products:
        # Keep the deterministic top product locked.
        locked_top_product = ranked_products[0]

        other_products = ranked_products[1:]

        if other_products:
            best_other_score = other_products[0]["score"]

            eligible_for_reranking = [
                item
                for item in other_products
                if item["score"]
                >= best_other_score - 3
            ]

            remaining_products = [
                item
                for item in other_products
                if item["score"]
                < best_other_score - 3
            ]

            eligible_for_reranking = (
                rerank_products_with_llm(
                    ranked_products=(
                        eligible_for_reranking
                    ),
                    signals=signals,
                    merged_profile=merged_profile,
                )
            )

            ranked_products = [
                locked_top_product,
                *eligible_for_reranking,
                *remaining_products,
            ]
    MIN_PRODUCT_SCORE = 6

    for ranked in ranked_products:
        if ranked["score"] < MIN_PRODUCT_SCORE:
            continue

        product_name = ranked["name"]
        product = ranked["product"]

        reason = build_product_reason(
            product_name=product_name,
            product=product,
            matched_signals=ranked["matched_signals"],
            language=language,
        )

        offer = get_best_offer(
            product_name,
            quantity_offers_db,
            bundle_offers_db,
        )
        loyalty_info = build_product_loyalty_message(
            user_code=(
                merged_profile.get("user_code")
                or merged_profile.get("user_id")
            ),
            product_name=product_name,
            language=merged_profile.get("language", "ar"),
        )
        decision["recommended_products"].append({
            "product": product_name,
            "reason": reason,
            "offer": offer,
            "price": product.get("price"),
            "currency": product.get("currency", "TND"),
            "image_emoji": product.get("image_emoji") or "🌿",
            "category": product.get("category"),
            "benefits": (product.get("benefits") or [])[:3],
            "matched_signals": ranked["matched_signals"],
            "score": ranked["score"],
            "loyalty_recommendation": loyalty_info,
        })
 


    # -------------------------
    if decision["recommended_products"]:
        decision["warnings"].append(
            translated_text(language, "supplement_warning")
        )
    
    selected_product_names = {
        item.get("product")
        for item in decision["recommended_products"]
    }

    if "Berberine & Ceylon Cinnamon" in selected_product_names:
        decision["warnings"].append(
            translated_text(
                language,
                "chronic_condition_warning",
            )
        )

    if "Lung Detox" in selected_product_names:
        decision["warnings"].append(
            translated_text(
                language,
                "persistent_symptoms_warning",
            )
        )
    
    conditions_text = " ".join(
        str(condition).lower()
        for condition in conditions
    )

    if (
        "diabetes" in conditions_text
        and "Berberine & Ceylon Cinnamon" in selected_product_names
    ):
        decision["warnings"].append(
            translated_text(language, "diabetes_warning")
        )

    if (
        "hypertension" in conditions_text
        and "Blood Detox" in selected_product_names
    ):
        decision["warnings"].append(
            translated_text(language, "hypertension_warning")
        )

    if priority_need in [
        "digestion",
        "bloating_gas",
        "constipation",
        "detox",
        "respiratory_support",
        "smoking_reduction_support",
        "glycemic_balance",
        "insulin_sensitivity",
        "blood_regulation",
        "heart_support",
        "stress_anxiety",
    ]:
        decision["warnings"].append(
            translated_text(
                language,
                "persistent_symptoms_warning",
            )
        )

    # -------------------------
    # 6. Upsell strategy
    # -------------------------
    best_bundle = find_best_bundle_for_recommendations(
        decision["recommended_products"],
        bundle_offers_db,
    )

    if best_bundle:

        decision["upsell_strategy"] = {
            "products": best_bundle.get("products", []),
            "offer": best_bundle,
            "message": translated_text(
                language,
                "best_offer",
            ),
        }

    elif decision["recommended_products"]:

        first_product = (
            decision["recommended_products"][0]["product"]
        )

        offer = get_best_offer(
            first_product,
            quantity_offers_db,
            bundle_offers_db,
        )

        if offer:

            decision["upsell_strategy"] = {
                "product": first_product,
                "offer": offer,
                "message": translated_text(
                    language,
                    "best_offer",
                ),
            }

    # -------------------------
    # 7. Confidence score
    # -------------------------
    confidence = 20

    if goals:
        confidence += 15

    if bmi:
        confidence += 10

    if conditions:
        confidence += 10

    if health_interests:
        confidence += 15

    if merged_profile.get("recent_checkins"):
        confidence += 15

    if merged_profile.get("recent_chat_history"):
        confidence += 5

    memory_has_useful_patterns = bool(
        merged_profile.get("past_recommended_products")
        or merged_profile.get("recurring_food_patterns")
        or merged_profile.get("recurring_activity_patterns")
        or merged_profile.get("last_detected_issue")
    )

    if memory_has_useful_patterns:
        confidence += 10

    decision["confidence_score"] = min(confidence, 95)

    return decision

def build_recommendation_agent_output(user_profile: dict):
    import time
    start_time = time.time()
    user_profile = user_profile or {}
    user_id = user_profile.get("user_id", "demo_user")
    memory = get_user_memory(user_id)
    profile = get_user_profile(user_id)
    recent_checkins = get_recent_checkins(user_id, limit=50)
    recent_steps = [
        c.get("daily_steps", 0)
        for c in recent_checkins
        if c.get("daily_steps") is not None
    ]

    average_steps_last_7_days = (
        sum(recent_steps) / len(recent_steps)
        if recent_steps
        else None
    )
    recent_chat_history = get_recent_chat_history(user_id, days=90, limit=100)
    quantity_offers_db = get_quantity_offers_dict()
    bundle_offers_db = get_bundle_offers()
    merged_profile = {
        "user_id": user_id,
        "age": user_profile.get("age") or profile.get("age"),
        "weight": user_profile.get("weight") or profile.get("weight"),
        "height": user_profile.get("height") or profile.get("height"),
        "goals": user_profile.get("goals") or profile.get("goals", []),
        "average_steps_last_7_days": average_steps_last_7_days,
        "medical_conditions": user_profile.get("medical_conditions") or profile.get("medical_conditions", []),
        "language": normalize_language(
            user_profile.get("language")
            or profile.get("language")
            or "ar"
        ),
        "sex": user_profile.get("sex") or profile.get("sex"),
   
        "trend_analysis": memory.get("trend_analysis"),
        "health_interests": memory.get("health_interests", []),
        "past_recommended_products": memory.get("past_recommended_products", []),
        "last_recommended_product": memory.get("last_recommended_product"),
        "recurring_food_patterns": memory.get("recurring_food_patterns", []),
        "recurring_activity_patterns": memory.get("recurring_activity_patterns", []),
        "last_meal_summary": memory.get("last_meal_summary"),
        "last_activity_summary": memory.get("last_activity_summary"),
        "last_detected_issue": memory.get("last_detected_issue"),
        "consistency_score": memory.get("consistency_score"),
        "notes": memory.get("notes", []),
        "recent_checkins": recent_checkins,
        "recent_chat_history": recent_chat_history,
        "recent_chat_topics": [
            {
                "question": c.get("question"),
                "intent": c.get("intent"),
                "detected_product": c.get("detected_product"),
                "recommended_product": c.get("recommended_product"),
            }
            for c in recent_chat_history
            if c.get("intent") not in ["greeting", "thanks", "small_talk", None]
        ],
        "recent_meals": [
            meal.get("description")
            for c in recent_checkins
            for meal in c.get("daily_meal_logs", [])
            if meal.get("description")
        ],
        "recent_activities": [
            activity.get("activity_type")
            for c in recent_checkins
            for activity in c.get("daily_activity_logs", [])
            if activity.get("activity_type")
        ],
        "recent_moods": [c.get("mood") for c in recent_checkins if c.get("mood")],
    }

    bmi = calculate_bmi(merged_profile.get("weight"), merged_profile.get("height"))
    merged_profile["bmi"] = bmi

    signals = extract_recommendation_signals_with_llm(merged_profile)
    decision = build_recommendation_decision(
        signals,
        merged_profile,
        bmi,
        quantity_offers_db,
        bundle_offers_db
    )
    nutrition_strategy = build_nutrition_strategy(
        merged_profile,
        signals
    )

    exercise_strategy = build_exercise_strategy(
        merged_profile,
        signals
    )
    dynamic_plan = generate_dynamic_plan_with_llm(
        merged_profile,
        signals,
        decision,
        nutrition_strategy,
        exercise_strategy
    )
    

    output = {
        "profile_summary": {
            "goals": merged_profile.get("goals"),
            "weight": merged_profile.get("weight"),
            "height": merged_profile.get("height"),
            "bmi": bmi,
            "age": merged_profile.get("age"),
            "medical_conditions": merged_profile.get("medical_conditions"),
            "sex": merged_profile.get("sex"),
            "average_steps_last_7_days": average_steps_last_7_days,
            "activity_level": signals.get("activity_level", "unknown"),
            "health_interests": signals.get("health_interests", []),
            "food_patterns": signals.get("food_patterns", []),
            "past_recommended_products": merged_profile.get("past_recommended_products", []),
            "last_recommended_product": merged_profile.get("last_recommended_product"),
            "last_meal_summary": merged_profile.get("last_meal_summary"),
            "last_activity_summary": merged_profile.get("last_activity_summary"),
            "last_detected_issue": merged_profile.get("last_detected_issue"),
            "consistency_score": merged_profile.get("consistency_score"),
            "recent_checkins_count": len(recent_checkins),
            "recent_meals": merged_profile.get("recent_meals"),
            "recent_activities": merged_profile.get("recent_activities"),
            "recent_chat_count": len(recent_chat_history),
            "recent_chat_topics": merged_profile.get("recent_chat_topics"),
        },

        "priority_focus": decision["priority_focus"],
        "recommended_products": decision["recommended_products"],

        "meal_recommendations": dynamic_plan.get("meal_recommendations", []),
        "exercise_recommendations": dynamic_plan.get("exercise_recommendations", []),
        "daily_actions": list({
            (
                item.get("action", ""),
                item.get("reason", ""),
            ): item
            for item in (
                decision.get("daily_actions", [])
                + dynamic_plan.get("daily_actions", [])
            )
        }.values()),
        "behavioral_insight": dynamic_plan.get("behavioral_insight", ""),
        "motivation_message": dynamic_plan.get("motivation_message", ""),
        "warnings": dynamic_plan.get("warnings", []) + decision.get("warnings", []),

        "upsell_strategy": decision["upsell_strategy"],
        "confidence_score": decision["confidence_score"],

        "reasoning_summary": (
            dynamic_plan.get("behavioral_insight")
            or translated_text(
                merged_profile.get("language", "ar"),
                "fallback_reasoning"
            )
        ),

        "llm_signals": signals,
        "decision_layer": decision
    }


    response_time = round(time.time() - start_time, 3)

    log_recommendation(user_id, {
        "recommended_products": [p.get("product") for p in output.get("recommended_products", [])],
        "meal_recommendations": output.get("meal_recommendations", []),
        "exercise_recommendations": output.get("exercise_recommendations", []),
        "reasoning_summary": output.get("reasoning_summary"),

        "used_memory": bool(memory.get("health_interests") or memory.get("past_recommended_products")),
        "used_goal": bool(merged_profile.get("goals")),
        "used_activity": (
            len(recent_checkins) > 0
        ),
        "used_bmi": bool(bmi),
        "used_daily_checkin": len(recent_checkins) > 0,
        "response_time_sec": response_time,
    })
    output["warnings"] = list(dict.fromkeys(output.get("warnings", [])))
    return output