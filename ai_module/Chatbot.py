import os
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
from ai_module.dashboard import (
    build_product_loyalty_message,
)
from ai_module.decision_engine import build_decision_output
from ai_module.user_memory_db import (
    get_user_memory,
    get_user_profile,
    update_user_memory,
    log_chat_interaction
)
from ai_module.recommendation_agent import (
    apply_digestive_product_priority,
    score_products,
    build_product_reason
)
from ai_module.product_db import (
    get_product_knowledge_dict,
    get_quantity_offers_dict,
    get_bundle_offers,
    get_all_products,
)  
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are Shifa's AI assistant, a professional nutrition and wellness assistant.

Your role:
- answer clearly about food, healthy habits, wellness, calories, and Shifa products
- use user profile if provided
- give practical and specific answers

LANGUAGE RULES:
- If the user writes in English, reply in English
- If the user writes in French, reply in French
- If the user writes in Arabic, reply in Arabic letters only
- If the user writes in Tunisian dialect, reply in Tunisian dialect using Arabic letters only
- If the user writes Tunisian dialect in Latin letters, convert your answer into Tunisian dialect written in Arabic letters
- Do not use Tunisian in Latin letters
- Do not mix Arabic letters and Latin-script Tunisian in the same answer

STRICT LANGUAGE RULE:
- For Arabic and Tunisian requests, your answer must be written only in Arabic script
- Do NOT answer Tunisian in Latin letters
- Do NOT switch to formal Arabic unless the user is clearly using formal Arabic
- If the user is Tunisian, prefer natural Tunisian darja in Arabic script

CONVERSATION MEMORY RULES:
- Use the previous conversation when the user refers to something implicitly
- If the user says things like "نستعملو", "هذا", "it", "this product", interpret them using recent chat history
- If the last discussed product is clear from the conversation, continue with that product
- Do not switch to another product unless the user explicitly changes the topic
PROFILE PRIORITY RULE:
- For nutrition, meals, calories, exercise, and healthy habit questions, always adapt the answer to the user's profile if available
- Especially use the user's goal, preferences, and activity information
- Do not give a generic answer if profile information is available
- If the user asks whether they can eat a certain food, answer according to their goal first

SCOPE RULE:
- You are only allowed to help with:
  1. Shifa products
  2. nutrition and meals
  3. calories
  4. exercise and activity
  5. healthy habits and wellness
- If the user asks about anything clearly outside these topics, do not answer the unrelated question.
- Instead, politely explain that you are Shifa's assistant and can help only with Shifa products, nutrition, calories, exercise, and wellness.
- Do not try to give partial answers for unrelated topics.

QUALITY RULES:
- Be clear and concise
- Give direct and concrete answers
- Do not give vague advice
- Do not invent facts
- Do not invent product information
- If information is missing, say clearly that you do not have it
- You must NEVER recommend or mention products that are not in the provided Shifa product knowledge base
- If no Shifa product matches the request, clearly say that no specific Shifa product is currently available for that goal

FALLBACK RULE:
- If the question is new or unclear, give a simple and general answer
- Do not invent stories
- Do not generate random names or meaningless sentences
- If unsure, give a short practical answer instead of guessing

SAFETY RULES:
- Do not give medical diagnosis
- Do not claim to cure diseases
- Do not replace a doctor or dietitian

- If the user has medical conditions, adapt advice safely.
- Do not diagnose or prescribe treatment.
- For diabetes, avoid recommending high-sugar meals or sugary drinks.
- For hypertension, avoid recommending high-salt or highly processed foods.
- For pregnancy, breastfeeding, chronic disease, or medication use, advise consulting a doctor or pharmacist before using supplements.

PRODUCT BOUNDARY RULES:
- You must NEVER mention or recommend products that are not موجودة في قاعدة معرفة Shifa
- Do not invent names like Protein Powder, Weight Gainer, or any external supplement



USAGE PRIORITY RULE:
- If the user asks how to use a product, answer first with the exact usage instructions
- Keep the answer direct and short
- Do not replace the usage instructions with general wellness advice

STYLE:
- Sound professional, helpful, and human
- Keep the answer direct and to the point
- Maximum 2 to 5 short sentences unless the user asks for details
- If product recommendation is relevant, mention it naturally
"""

TUNISIAN_EXAMPLES = """
Example:
Q: شنوة ناكل اليوم باش ننقص وزن؟
A:
حسب هدفك، تنجم تختار وجبات خفيفة ومتوازنة:
- فطور فيه بروتين + شوية غلة
- غدا فيه دجاج ولا حوت + سلطة
- عشا حاجة خفيفة كيف شوربة ولا سلطة
"""

BRAND_INFO = """
Brand: Shifa
Description: Shifa is a wellness and supplement brand.
General rule: Shifa products are dietary supplements, not medications.
They support a healthy lifestyle and do not replace medical advice.
"""
def get_product_knowledge_dict():
    products = get_all_products()
    return {
        product["name"]: product
        for product in products
        if product.get("name")
    }




def build_user_context(merged_profile: dict | None) -> str:
    if not merged_profile:
        return "No user profile provided."

    return f"""
User profile:
- age: {merged_profile.get('age')}
- weight: {merged_profile.get('weight')}
- height: {merged_profile.get('height')}
- goals: {merged_profile.get('goals')}
- medical_conditions: {merged_profile.get('medical_conditions')}
- activity_info: {merged_profile.get('activity_info')}
- sex: {merged_profile.get('sex')}
Daily check-in memory:
- health_interests: {merged_profile.get('health_interests')}
- recurring_food_patterns: {merged_profile.get('recurring_food_patterns')}
- recurring_activity_patterns: {merged_profile.get('recurring_activity_patterns')}
- last_meal_summary: {merged_profile.get('last_meal_summary')}
- last_activity_summary: {merged_profile.get('last_activity_summary')}
- last_detected_issue: {merged_profile.get('last_detected_issue')}
- consistency_score: {merged_profile.get('consistency_score')}
"""


def normalize_text(value: str | None) -> str:
    return " ".join(
        str(value or "").lower().strip().split()
    )


def detect_product(
    question: str,
    product_db: dict | None = None,
) -> str | None:
    """
    Detect only a product explicitly mentioned by name.

    Generic needs such as weight loss, appetite, digestion,
    sugar balance, etc. must not count as an explicit product.
    """
    q = normalize_text(question)

    product_db = (
        product_db
        or get_product_knowledge_dict()
    )

    for product_name in product_db:
        normalized_name = normalize_text(product_name)

        if normalized_name and normalized_name in q:
            return product_name

    return None

def detect_products(
    question: str,
    product_db: dict | None = None,
) -> list[str]:
    q = normalize_text(question)

    product_db = (
        product_db
        or get_product_knowledge_dict()
    )

    detected = []

    product_names = sorted(
        product_db.keys(),
        key=len,
        reverse=True,
    )

    for product_name in product_names:
        normalized_name = normalize_text(
            product_name
        )

        if normalized_name and normalized_name in q:
            detected.append(product_name)

    return list(dict.fromkeys(detected))


# FIX: detect_product/detect_products exigent soit le nom exact du
# produit, soit un tag explicite dans product_db -- un terme de
# categorie generique ("produit detox", "produit minceur") ne matche
# aucun des deux car DEUX produits ou plus partagent ce mot (Liver
# Detox et Colon Detox contiennent tous les deux "Detox"), donc aucun
# match n'est unique et detect_product renvoie None. Ce helper detecte
# ces cas d'ambiguite pour permettre au chatbot de demander une
# precision au lieu de repondre dans le vide.
CATEGORY_KEYWORDS: dict[str, set[str]] = {
    # ==========================================================
    # GENERAL DETOX — ambiguous category
    # ==========================================================
    "detox": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "détox": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "detoxification": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "produit detox": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "cure detox": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "تنظيف الجسم": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "تنقية الجسم": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "التخلص من السموم": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "ديتوكس": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "نحب نعمل ديتوكس": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "nheb naamel detox": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },
    "tandhif el jism": {
        "Colon Detox",
        "Liver Detox",
        "Blood Detox",
        "Lung Detox",
    },

    # ==========================================================
    # COLON DETOX
    # ==========================================================
    "colon": {"Colon Detox"},
    "côlon": {"Colon Detox"},
    "colon detox": {"Colon Detox"},
    "détox du côlon": {"Colon Detox"},
    "nettoyage du côlon": {"Colon Detox"},
    "nettoyer le côlon": {"Colon Detox"},
    "santé du côlon": {"Colon Detox"},
    "confort du côlon": {"Colon Detox"},
    "digestion difficile": {"Colon Detox"},
    "mauvaise digestion": {"Colon Detox"},
    "inconfort digestif": {"Colon Detox"},
    "troubles digestifs": {"Colon Detox"},
    "ballonnement": {"Colon Detox"},
    "ballonnements": {"Colon Detox"},
    "ventre gonflé": {"Colon Detox"},
    "gaz": {"Colon Detox"},
    "gaz intestinaux": {"Colon Detox"},
    "flatulence": {"Colon Detox"},
    "flatulences": {"Colon Detox"},
    "lourdeur digestive": {"Colon Detox"},
    "transit lent": {"Colon Detox", "Psyllium"},
    "constipation": {"Colon Detox", "Psyllium"},

    "القولون": {"Colon Detox"},
    "تنظيف القولون": {"Colon Detox"},
    "تطهير القولون": {"Colon Detox"},
    "صحة القولون": {"Colon Detox"},
    "مشاكل القولون": {"Colon Detox"},
    "الهضم": {"Colon Detox"},
    "سوء الهضم": {"Colon Detox"},
    "عسر الهضم": {"Colon Detox"},
    "مشاكل الهضم": {"Colon Detox"},
    "انتفاخ": {"Colon Detox"},
    "انتفاخ البطن": {"Colon Detox"},
    "نفخة": {"Colon Detox"},
    "غازات": {"Colon Detox"},
    "غازات البطن": {"Colon Detox"},
    "ثقل في المعدة": {"Colon Detox"},
    "معدة ثقيلة": {"Colon Detox"},

    "kerch": {"Colon Detox"},
    "kerchi": {"Colon Detox"},
    "nfekh": {"Colon Detox"},
    "nefkh": {"Colon Detox"},
    "naf5a": {"Colon Detox"},
    "gazet": {"Colon Detox"},
    "ghazet": {"Colon Detox"},
    "m3adti": {"Colon Detox"},
    "maadti": {"Colon Detox"},
    "m3adti th9ila": {"Colon Detox"},
    "hadhma": {"Colon Detox"},
    "hadma": {"Colon Detox"},
    "mochklet hadhm": {"Colon Detox"},
    "li ynadhef lcolon": {"Colon Detox"},
    "li ynadhaf lcolon": {"Colon Detox"},
    "nadhafli lcolon": {"Colon Detox"},
    "ynadhef el colon": {"Colon Detox"},

    # ==========================================================
    # PSYLLIUM
    # ==========================================================
    "psyllium": {"Psyllium"},
    "psilium": {"Psyllium"},
    "ispaghul": {"Psyllium"},
    "tégument de psyllium": {"Psyllium"},
    "fibre": {"Psyllium"},
    "fibres": {"Psyllium"},
    "fibres alimentaires": {"Psyllium"},
    "complément de fibres": {"Psyllium"},
    "manque de fibres": {"Psyllium"},
    "transit intestinal": {"Psyllium"},
    "régularité intestinale": {"Psyllium"},
    "aller aux toilettes": {"Psyllium"},
    "selles difficiles": {"Psyllium"},
    "constipation occasionnelle": {"Psyllium"},
    "satiété": {"Psyllium"},
    "coupe faim naturel": {"Psyllium"},
    "contrôle de l'appétit": {"Psyllium"},
    "réduire l'appétit": {"Psyllium"},
    "manger moins": {"Psyllium"},

    "السيليوم": {"Psyllium"},
    "بسيليوم": {"Psyllium"},
    "قشور السيليوم": {"Psyllium"},
    "ألياف": {"Psyllium"},
    "الألياف": {"Psyllium"},
    "نقص الألياف": {"Psyllium"},
    "تنظيم الأمعاء": {"Psyllium"},
    "حركة الأمعاء": {"Psyllium"},
    "تسهيل الإخراج": {"Psyllium"},
    "صعوبة الإخراج": {"Psyllium"},
    "إمساك": {"Psyllium", "Colon Detox"},
    "امساك": {"Psyllium", "Colon Detox"},
    "الشبع": {"Psyllium"},
    "زيادة الشبع": {"Psyllium"},
    "تقليل الشهية": {"Psyllium"},
    "التحكم في الشهية": {"Psyllium"},

    "imsek": {"Psyllium", "Colon Detox"},
    "emsek": {"Psyllium", "Colon Detox"},
    "emsak": {"Psyllium", "Colon Detox"},
    "ma nemchich lel toilette": {"Psyllium"},
    "s3ib nemchi lel toilette": {"Psyllium"},
    "s3ib nokhrej": {"Psyllium"},
    "nheb fibre": {"Psyllium"},
    "nheb nechba3": {"Psyllium"},
    "ma nechba3ch": {"Psyllium"},
    "chahiya kbira": {"Psyllium"},
    "yn9as chahiya": {"Psyllium"},

    # ==========================================================
    # LIVER DETOX
    # ==========================================================
    "liver detox": {"Liver Detox"},
    "liver": {"Liver Detox"},
    "foie": {"Liver Detox"},
    "détox du foie": {"Liver Detox"},
    "nettoyage du foie": {"Liver Detox"},
    "nettoyer le foie": {"Liver Detox"},
    "santé du foie": {"Liver Detox"},
    "soutien du foie": {"Liver Detox"},
    "fonction hépatique": {"Liver Detox"},
    "fonctionnement du foie": {"Liver Detox"},
    "hépatique": {"Liver Detox"},
    "bile": {"Liver Detox"},
    "sécrétion biliaire": {"Liver Detox"},

    "الكبد": {"Liver Detox"},
    "تنظيف الكبد": {"Liver Detox"},
    "ديتوكس الكبد": {"Liver Detox"},
    "تطهير الكبد": {"Liver Detox"},
    "صحة الكبد": {"Liver Detox"},
    "دعم الكبد": {"Liver Detox"},
    "وظائف الكبد": {"Liver Detox"},
    "الصفراء": {"Liver Detox"},
    "العصارة الصفراوية": {"Liver Detox"},

    "kebda": {"Liver Detox"},
    "kabda": {"Liver Detox"},
    "kebdi": {"Liver Detox"},
    "nadhaf el kebda": {"Liver Detox"},
    "ynadhef lkebda": {"Liver Detox"},
    "detox kebda": {"Liver Detox"},
    "produit lel kebda": {"Liver Detox"},
    "haja lel kebda": {"Liver Detox"},

    # ==========================================================
    # SLIM PACK
    # ==========================================================
    "slim pack": {"Slim Pack"},
    "slim day": {"Slim Pack"},
    "slim night": {"Slim Pack"},
    "pack minceur": {"Slim Pack"},
    "produit minceur": {"Slim Pack"},
    "complément minceur": {"Slim Pack"},
    "minceur": {"Slim Pack"},
    "amaigrissement": {"Slim Pack"},
    "perte de poids": {"Slim Pack"},
    "perdre du poids": {"Slim Pack"},
    "réduction du poids": {"Slim Pack"},
    "gestion du poids": {"Slim Pack"},
    "contrôle du poids": {"Slim Pack"},
    "surpoids": {"Slim Pack"},
    "excès de poids": {"Slim Pack"},
    "brûleur de graisse": {"Slim Pack"},
    "bruleur de graisse": {"Slim Pack"},
    "graisse corporelle": {"Slim Pack"},
    "graisse abdominale": {"Slim Pack"},
    "métabolisme": {"Slim Pack"},
    "activer le métabolisme": {"Slim Pack"},
    "appétit": {"Slim Pack", "Psyllium"},
    "réduire la faim": {"Slim Pack", "Psyllium"},

    "التنحيف": {"Slim Pack"},
    "تخسيس": {"Slim Pack"},
    "إنقاص الوزن": {"Slim Pack"},
    "نقص الوزن": {"Slim Pack"},
    "فقدان الوزن": {"Slim Pack"},
    "التحكم في الوزن": {"Slim Pack"},
    "الوزن الزائد": {"Slim Pack"},
    "الوزن الزائد": {"Slim Pack"},
    "السمنة": {"Slim Pack"},
    "حرق الدهون": {"Slim Pack"},
    "دهون البطن": {"Slim Pack"},
    "الشحوم": {"Slim Pack"},
    "الكرش": {"Slim Pack"},
    "الأيض": {"Slim Pack"},
    "رفع معدل الحرق": {"Slim Pack"},
    "تقليل الجوع": {"Slim Pack", "Psyllium"},

    "na9es wazn": {"Slim Pack"},
    "naqs wazn": {"Slim Pack"},
    "n9as lwazn": {"Slim Pack"},
    "nheb nna9es": {"Slim Pack"},
    "nheb nodh3ef": {"Slim Pack"},
    "nheb nodh3of": {"Slim Pack"},
    "nheb ndha3ef": {"Slim Pack"},
    "ydha3ef": {"Slim Pack"},
    "li ydha3ef": {"Slim Pack"},
    "li yna9es lwazn": {"Slim Pack"},
    "ynaqes lwazn": {"Slim Pack"},
    "yn9as lwazn": {"Slim Pack"},
    "takhssis": {"Slim Pack"},
    "tan7if": {"Slim Pack"},
    "semna": {"Slim Pack"},
    "smen": {"Slim Pack"},
    "graisse": {"Slim Pack"},
    "ch7am": {"Slim Pack"},
    "kerch kbira": {"Slim Pack"},
    "metabolisme": {"Slim Pack"},
    "har9 edhoun": {"Slim Pack"},

    # ==========================================================
    # BLOOD DETOX
    # ==========================================================
    "blood detox": {"Blood Detox"},
    "sang": {"Blood Detox"},
    "détox du sang": {"Blood Detox"},
    "nettoyer le sang": {"Blood Detox"},
    "purifier le sang": {"Blood Detox"},
    "purification du sang": {"Blood Detox"},
    "circulation": {"Blood Detox"},
    "circulation sanguine": {"Blood Detox"},
    "mauvaise circulation": {"Blood Detox"},
    "santé cardiovasculaire": {"Blood Detox"},
    "santé du cœur": {"Blood Detox"},
    "coeur": {"Blood Detox"},
    "cœur": {"Blood Detox"},
    "cardiovasculaire": {"Blood Detox"},

    "الدم": {"Blood Detox"},
    "تنقية الدم": {"Blood Detox"},
    "تنظيف الدم": {"Blood Detox"},
    "تطهير الدم": {"Blood Detox"},
    "الدورة الدموية": {"Blood Detox"},
    "ضعف الدورة الدموية": {"Blood Detox"},
    "صحة القلب": {"Blood Detox"},
    "القلب": {"Blood Detox"},
    "دعم القلب": {"Blood Detox"},
    "الأوعية الدموية": {"Blood Detox"},

    "dam": {"Blood Detox"},
    "damm": {"Blood Detox"},
    "ysafi dam": {"Blood Detox"},
    "li ysafi dam": {"Blood Detox"},
    "ynadhef dam": {"Blood Detox"},
    "dawra damawiya": {"Blood Detox"},
    "dawran eddam": {"Blood Detox"},
    "circulation dam": {"Blood Detox"},
    "9alb": {"Blood Detox"},
    "qalb": {"Blood Detox"},
    "produit lel 9alb": {"Blood Detox"},

    # ==========================================================
    # LUNG DETOX
    # ==========================================================
    "lung detox": {"Lung Detox"},
    "poumon": {"Lung Detox"},
    "poumons": {"Lung Detox"},
    "détox des poumons": {"Lung Detox"},
    "nettoyer les poumons": {"Lung Detox"},
    "santé pulmonaire": {"Lung Detox"},
    "santé respiratoire": {"Lung Detox"},
    "confort respiratoire": {"Lung Detox"},
    "respiration": {"Lung Detox"},
    "respirer": {"Lung Detox"},
    "difficulté à respirer": {"Lung Detox"},
    "fumeur": {"Lung Detox"},
    "fumeuse": {"Lung Detox"},
    "tabac": {"Lung Detox"},
    "cigarette": {"Lung Detox"},
    "réduire le tabac": {"Lung Detox"},
    "arrêter de fumer": {"Lung Detox"},
    "réduction du tabagisme": {"Lung Detox"},
    "rweri": {"Lung Detox"},

    "الرئة": {"Lung Detox"},
    "الرئتين": {"Lung Detox"},
    "تنظيف الرئة": {"Lung Detox"},
    "تنظيف الرئتين": {"Lung Detox"},
    "صحة الرئة": {"Lung Detox"},
    "الجهاز التنفسي": {"Lung Detox"},
    "التنفس": {"Lung Detox"},
    "صعوبة التنفس": {"Lung Detox"},
    "ضيق التنفس": {"Lung Detox"},
    "التدخين": {"Lung Detox"},
    "مدخن": {"Lung Detox"},
    "مدخنة": {"Lung Detox"},
    "السجائر": {"Lung Detox"},
    "تقليل التدخين": {"Lung Detox"},
    "الإقلاع عن التدخين": {"Lung Detox"},

    "riya": {"Lung Detox"},
    "rya": {"Lung Detox"},
    "raya": {"Lung Detox"},
    "poumonet": {"Lung Detox"},
    "tanaffos": {"Lung Detox"},
    "nafas": {"Lung Detox"},
    "ma najamch netnaffes": {"Lung Detox"},
    "netnaffes bs3ouba": {"Lung Detox"},
    "dokhan": {"Lung Detox"},
    "doukhan": {"Lung Detox"},
    "cigarette": {"Lung Detox"},
    "cigaro": {"Lung Detox"},
    "nheb nna9es dokhan": {"Lung Detox"},
    "nheb nbatal dokhan": {"Lung Detox"},
    "nheb nbatel dokhan": {"Lung Detox"},

    # ==========================================================
    # BERBERINE & CEYLON CINNAMON
    # ==========================================================
    "berberine": {"Berberine & Ceylon Cinnamon"},
    "berbérine": {"Berberine & Ceylon Cinnamon"},
    "ceylon cinnamon": {"Berberine & Ceylon Cinnamon"},
    "cannelle de ceylan": {"Berberine & Ceylon Cinnamon"},
    "cannelle": {"Berberine & Ceylon Cinnamon"},
    "équilibre glycémique": {"Berberine & Ceylon Cinnamon"},
    "glycémie": {"Berberine & Ceylon Cinnamon"},
    "taux de sucre": {"Berberine & Ceylon Cinnamon"},
    "sucre dans le sang": {"Berberine & Ceylon Cinnamon"},
    "glucose": {"Berberine & Ceylon Cinnamon"},
    "insuline": {"Berberine & Ceylon Cinnamon"},
    "sensibilité à l'insuline": {"Berberine & Ceylon Cinnamon"},
    "résistance à l'insuline": {"Berberine & Ceylon Cinnamon"},
    "envie de sucre": {"Berberine & Ceylon Cinnamon"},
    "envies de sucre": {"Berberine & Ceylon Cinnamon"},
    "envie de sucré": {"Berberine & Ceylon Cinnamon"},
    "fringale sucrée": {"Berberine & Ceylon Cinnamon"},
    "diabète": {"Berberine & Ceylon Cinnamon"},
    "diabétique": {"Berberine & Ceylon Cinnamon"},

    "البربرين": {"Berberine & Ceylon Cinnamon"},
    "بربرين": {"Berberine & Ceylon Cinnamon"},
    "قرفة سيلان": {"Berberine & Ceylon Cinnamon"},
    "القرفة": {"Berberine & Ceylon Cinnamon"},
    "توازن السكر": {"Berberine & Ceylon Cinnamon"},
    "تنظيم السكر": {"Berberine & Ceylon Cinnamon"},
    "سكر الدم": {"Berberine & Ceylon Cinnamon"},
    "مستوى السكر": {"Berberine & Ceylon Cinnamon"},
    "الجلوكوز": {"Berberine & Ceylon Cinnamon"},
    "الإنسولين": {"Berberine & Ceylon Cinnamon"},
    "مقاومة الإنسولين": {"Berberine & Ceylon Cinnamon"},
    "حساسية الإنسولين": {"Berberine & Ceylon Cinnamon"},
    "الرغبة في السكر": {"Berberine & Ceylon Cinnamon"},
    "الرغبة في الحلويات": {"Berberine & Ceylon Cinnamon"},
    "اشتهاء الحلويات": {"Berberine & Ceylon Cinnamon"},
    "السكري": {"Berberine & Ceylon Cinnamon"},

    "qerfa": {"Berberine & Ceylon Cinnamon"},
    "9erfa": {"Berberine & Ceylon Cinnamon"},
    "korfa": {"Berberine & Ceylon Cinnamon"},
    "sokker": {"Berberine & Ceylon Cinnamon"},
    "souker": {"Berberine & Ceylon Cinnamon"},
    "سكر": {"Berberine & Ceylon Cinnamon"},
    "sokker fel dam": {"Berberine & Ceylon Cinnamon"},
    "taux sokker": {"Berberine & Ceylon Cinnamon"},
    "insuline": {"Berberine & Ceylon Cinnamon"},
    "mo9awmet insuline": {"Berberine & Ceylon Cinnamon"},
    "nchahi lel 7low": {"Berberine & Ceylon Cinnamon"},
    "nchahi lel hlou": {"Berberine & Ceylon Cinnamon"},
    "nheb l7low": {"Berberine & Ceylon Cinnamon"},
    "chahiya lel sokker": {"Berberine & Ceylon Cinnamon"},
}

def resolve_products_by_category(
    question: str,
    product_db: dict | None = None,
) -> tuple[list[str], list[list[str]]]:
    """
    Returns:
    - resolved_products: products coming from clear categories
    - ambiguous_groups: categories that correspond to several products
    """
    product_db = (
        product_db
        or get_product_knowledge_dict()
    )

    q = normalize_text(question)

    resolved_products = []
    ambiguous_groups = []

    sorted_keywords = sorted(
        CATEGORY_KEYWORDS,
        key=len,
        reverse=True,
    )

    matched_spans = []

    for keyword in sorted_keywords:
        normalized_keyword = normalize_text(keyword)

        if not normalized_keyword:
            continue

        if normalized_keyword not in q:
            continue

        # Avoid counting a short keyword already contained
        # inside a more specific matched expression.
        if any(
            normalized_keyword in previous_keyword
            for previous_keyword in matched_spans
        ):
            continue

        matched_spans.append(normalized_keyword)

        existing_products = [
            product_name
            for product_name in CATEGORY_KEYWORDS[keyword]
            if product_name in product_db
        ]

        if len(existing_products) == 1:
            resolved_products.extend(
                existing_products
            )

        elif len(existing_products) > 1:
            ambiguous_groups.append(
                existing_products
            )

    return (
        list(dict.fromkeys(resolved_products)),
        ambiguous_groups,
    )

def match_products_by_category(
    question: str,
    product_db: dict | None = None,
) -> list[str]:
    product_db = product_db or get_product_knowledge_dict()
    q = normalize_text(question)

    matched_products = set()

    # Longer phrases first, so specific expressions are prioritized.
    sorted_keywords = sorted(
        CATEGORY_KEYWORDS,
        key=len,
        reverse=True,
    )

    for keyword in sorted_keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword and normalized_keyword in q:
            matched_products.update(
                CATEGORY_KEYWORDS[keyword]
            )

    # Return only products that actually exist in Supabase/product_db.
    return [
        product_name
        for product_name in product_db
        if product_name in matched_products
    ]



def fuzzy_match_products_by_category(
    question: str,
    product_db: dict | None = None,
    threshold: int = 88,
) -> list[str]:
    product_db = product_db or get_product_knowledge_dict()

    q_words = normalize_text(question).split()
    matched_products = set()

    for keyword, products in CATEGORY_KEYWORDS.items():
        keyword_words = normalize_text(keyword).split()

        # Compare complete short expressions
        if len(keyword_words) > 1:
            score = fuzz.partial_ratio(
                normalize_text(question),
                normalize_text(keyword),
            )

            if score >= threshold:
                matched_products.update(products)

        # Compare individual words such as reya / riya
        else:
            for word in q_words:
                score = fuzz.ratio(
                    word,
                    keyword_words[0],
                )

                if score >= threshold:
                    matched_products.update(products)

    return [
        product_name
        for product_name in product_db
        if product_name in matched_products
    ]

def fuzzy_category_match(question, threshold=88):
    q_words = normalize_text(question).split()

    matched = set()

    for keyword, products in CATEGORY_KEYWORDS.items():
        for word in q_words:
            if fuzz.ratio(word, keyword) >= threshold:
                matched.update(products)

    return list(matched)
def build_disambiguation_answer(question: str, category_matches: list[str]) -> str:
    language = detect_fallback_language(question)
    names = ", ".join(category_matches)

    templates = {
        "ar": f"عندنا كذا منتج يلزم هالموضوع: {names}. شنية بالضبط تحب تعرف عليه؟",
        "fr": f"Nous avons plusieurs produits qui correspondent : {names}. Lequel vous intéresse ?",
        "en": f"We have a few products that match: {names}. Which one would you like to know about?",
    }

    return templates.get(language, templates["fr"])
def get_last_category_matches_from_history(
    chat_history: list | None,
    product_db: dict | None = None,
) -> list[str]:
    if not chat_history:
        return []

    for msg in reversed(chat_history):
        content = msg.get("content", "")
        matches = match_products_by_category(content, product_db)
        if matches:
            return matches

    return []


def format_delivery_info_for_products(products: list, quantity_offers_db: dict, bundle_offers_db: list) -> str:
    if not products:
        return ""

    lines = []

    # 1) Check if there is a bundle offer for these products together
    selected = set(products)

    matching_bundles = []
    for bundle in bundle_offers_db:
        bundle_products = set(bundle.get("products", []))

        if selected.issubset(bundle_products) or bundle_products.issubset(selected):
            matching_bundles.append(bundle)

    if matching_bundles:
        lines.append("عندك زادة عرض pack يجمع المنتجات هاذم:")

        for bundle in matching_bundles:
            fee = bundle.get("delivery_fee")
            delivery = "التوصيل مجاني" if fee == 0 else f"التوصيل {fee} د.ت"

            lines.append(
                f"- {bundle.get('title')}: "
                f"{bundle.get('new_price')} {bundle.get('currency')} "
                f"بدل {bundle.get('old_price')} {bundle.get('currency')}، "
                f"{delivery}."
            )

        lines.append("العرض هذا أنفع من شراء كل منتج وحدو.")

    # 2) Then show delivery per product/quantity
    for product in products:
        lines.append(f"\n{product}:")

        offers = quantity_offers_db.get(product, [])

        if offers:
            for offer in offers:
                title = offer.get("title") or f"{offer.get('quantity')} علبة"
                delivery = offer.get("delivery_text")

                if not delivery and offer.get("delivery_fee") is not None:
                    fee = offer.get("delivery_fee")
                    delivery = "التوصيل مجاني" if fee == 0 else f"التوصيل {fee} د.ت"

                lines.append(f"- {title}: {delivery}")
        else:
            lines.append("- ما عنديش تفاصيل توصيل حسب الكمية لهذا المنتج وحدو.")

    return "\n".join(lines)
def get_products_context(
    detected_products,
    product_db,
):
    products = []

    for name in detected_products:
        product = get_product_safely(
            product_db,
            name,
        )

        if product:
            products.append(product)

    return products

def format_multi_product_prices(
    products: list[str],
    product_db: dict,
    bundle_offers_db: list[dict],
    language: str = "ar",
) -> tuple[str, list[dict], list[dict]]:
    product_prices = []

    for product_name in products:
        product = get_product_safely(
            product_db,
            product_name,
        )

        if not product:
            continue

        product_prices.append({
            "product": product_name,
            "price": product.get("price"),
            "old_price": product.get("old_price"),
            "currency": (
                product.get("currency")
                or "TND"
            ),
        })

    requested_products = set(products)
    matching_bundles = []

    for bundle in bundle_offers_db:
        bundle_products = set(
            bundle.get("products") or []
        )

        # The bundle must contain all requested products
        if (
            requested_products
            and requested_products.issubset(
                bundle_products
            )
        ):
            matching_bundles.append(bundle)

    if language == "fr":
        lines = []

        for item in product_prices:
            price = item.get("price")
            old_price = item.get("old_price")
            currency = item.get("currency")

            if price is None:
                lines.append(
                    f"• Le prix de {item['product']} "
                    f"n’est pas disponible actuellement."
                )
            elif old_price and old_price > price:
                lines.append(
                    f"• {item['product']} : "
                    f"{price:g} {currency} "
                    f"au lieu de {old_price:g} {currency}."
                )
            else:
                lines.append(
                    f"• {item['product']} : "
                    f"{price:g} {currency}."
                )

        if matching_bundles:
            lines.append(
                "\nIls sont également disponibles ensemble :"
            )

            for bundle in matching_bundles:
                title = (
                    bundle.get("title")
                    or bundle.get("name")
                    or "Pack"
                )
                new_price = bundle.get("new_price")
                old_price = bundle.get("old_price")
                currency = (
                    bundle.get("currency")
                    or "TND"
                )

                if new_price is not None:
                    if (
                        old_price is not None
                        and old_price > new_price
                    ):
                        lines.append(
                            f"• {title} : "
                            f"{new_price:g} {currency} "
                            f"au lieu de "
                            f"{old_price:g} {currency}."
                        )
                    else:
                        lines.append(
                            f"• {title} : "
                            f"{new_price:g} {currency}."
                        )

        return (
            "\n".join(lines),
            product_prices,
            matching_bundles,
        )

    if language == "en":
        lines = []

        for item in product_prices:
            price = item.get("price")
            old_price = item.get("old_price")
            currency = item.get("currency")

            if price is None:
                lines.append(
                    f"• The current price of "
                    f"{item['product']} is unavailable."
                )
            elif old_price and old_price > price:
                lines.append(
                    f"• {item['product']}: "
                    f"{price:g} {currency} instead of "
                    f"{old_price:g} {currency}."
                )
            else:
                lines.append(
                    f"• {item['product']}: "
                    f"{price:g} {currency}."
                )

        if matching_bundles:
            lines.append(
                "\nThey are also available together:"
            )

            for bundle in matching_bundles:
                title = (
                    bundle.get("title")
                    or bundle.get("name")
                    or "Bundle"
                )
                new_price = bundle.get("new_price")
                old_price = bundle.get("old_price")
                currency = (
                    bundle.get("currency")
                    or "TND"
                )

                if new_price is not None:
                    if (
                        old_price is not None
                        and old_price > new_price
                    ):
                        lines.append(
                            f"• {title}: "
                            f"{new_price:g} {currency} "
                            f"instead of "
                            f"{old_price:g} {currency}."
                        )
                    else:
                        lines.append(
                            f"• {title}: "
                            f"{new_price:g} {currency}."
                        )

        return (
            "\n".join(lines),
            product_prices,
            matching_bundles,
        )

    # Arabic / Tunisian
    lines = []

    for item in product_prices:
        price = item.get("price")
        old_price = item.get("old_price")

        if price is None:
            lines.append(
                f"• سعر {item['product']} "
                f"موش متوفر حالياً."
            )
        elif old_price and old_price > price:
            lines.append(
                f"• {item['product']}: "
                f"{price:g} دينار عوض "
                f"{old_price:g} دينار."
            )
        else:
            lines.append(
                f"• {item['product']}: "
                f"{price:g} دينار."
            )

    if matching_bundles:
        lines.append(
            "\nموجودين زادة مع بعضهم في عرض:"
        )

        for bundle in matching_bundles:
            title = (
                bundle.get("title")
                or bundle.get("name")
                or "باك"
            )
            new_price = bundle.get("new_price")
            old_price = bundle.get("old_price")

            if new_price is not None:
                if (
                    old_price is not None
                    and old_price > new_price
                ):
                    lines.append(
                        f"• {title}: "
                        f"{new_price:g} دينار عوض "
                        f"{old_price:g} دينار."
                    )
                else:
                    lines.append(
                        f"• {title}: "
                        f"{new_price:g} دينار."
                    )

    return (
        "\n".join(lines),
        product_prices,
        matching_bundles,
    )

def get_last_product_from_history(chat_history: list | None) -> str | None:
    if not chat_history:
        return None

    for msg in reversed(chat_history):
        content = msg.get("content", "")
        detected = detect_product(content)
        if detected:
            return detected

    return None


def is_implicit_reference(question: str) -> bool:
    q = question.lower()
    keywords = [
        "نستعملو",
        "نستعمل",
        "كيفاش نستعمل",
        "kifeh nestaamlou",
        "nestaamlou",
        "how to use it",
        "how long use it",
        "use it",
        "hedha",
        "this product",
        "ce produit",
        "قداش من شهر",
        "قداش مدة",
        "how long",
        "for how long",
    ]
    return any(k in q for k in keywords)
def get_localized_field(
    product: dict,
    field: str,
    language: str,
):
    supported_languages = {"ar", "fr", "en"}

    if language not in supported_languages:
        language = "en"

    localized_value = product.get(
        f"{field}_{language}"
    )

    if localized_value:
        return localized_value

    return (
        product.get(f"{field}_en")
        or product.get(field)
        or []
    )
def format_product_field_for_many(
    products,
    field,
    title,
    language,
):
    sections = []
    structured_info = []

    for product in products:
        product_name = product.get(
            "name",
            "Produit",
        )

        value = get_localized_field(
            product,
            field,
            language,
        )

        if not value:
            continue

        if isinstance(value, list):
            formatted_value = "\n".join(
                f"- {item}"
                for item in value
            )
        else:
            formatted_value = str(value)

        sections.append(
            f"{title} {product_name}:\n"
            f"{formatted_value}"
        )

        structured_info.append({
            "product": product_name,
            field: value,
        })

    answer = (
        "\n\n------------------\n\n".join(
            sections
        )
    )

    return answer, structured_info
def detect_intent(question: str) -> str:
    q = question.lower()
    product = detect_product(question)
    if any(w in q for w in [
        "نفخة", "غازات", "كرشي", "معدة", "هضم", "امساك", "إمساك",
        "bloating", "gas", "constipation", "digestion", "nfekh", "gaz"
    ]):
        return "digestion_issue"

    if any(w in q for w in [
        "متقلق", "قلق", "stress", "stressed", "anxiety",
        "مانجمش نرقد", "ما نرقدش", "نوم", "sleep", "insomnia"
    ]):
        return "stress_sleep_issue"

    if any(word in q for word in ["hi", "hello", "aslema", "slm", "salem", "bonjour"]):
        return "greeting"

    
    if any(word in q for word in [
        "livraison", "delivery", "shipping", "frais livraison",
        "توصيل", "التوصيل", "قداش التوصيل", "توصل", "livrer"
    ]):
        return "delivery_info"

    if any(word in q for word in [
        "قداش", "سعر", "ثمن", "bqadeh", "prix", "price", "soum", "how much", "combien", "bgideh"
    ]):
        return "price_offer_query"

    if any(word in q for word in [
        "عرض", "promo", "promotion", "offer"
    ]):

        return "price_offer_query"
    # FIX: l'intent ingredients/benefits/usage est desormais detecte
    # independamment de la resolution exacte du produit (avant, ces 3
    # branches exigeaient `product` non-nul -- une question du type
    # "ingredients du produit minceur" (terme generique, pas un nom
    # exact) faisait echouer detect_product, donc la question tombait
    # dans product_recommendation au lieu de product_ingredients, meme
    # si le mot "ingredients" etait bien present). La reponse en aval
    # doit gerer le cas ou product est None en demandant une precision
    # ("lequel de nos produits ?") plutot que de repondre dans le vide.
    if any(word in q for word in [
        "ingredient",
        "ingredients",
        "ingredients",
        "mokawnet",
        "ingrédients",
        "composition",
        "composants",
        "تركيبة",
        "مكونات",
        "شنوة فيه",
        "chneya fih",
        "chnowa fih",
    ]):
        return "product_ingredients"

    if any(word in q for word in ["faida", "faidet", "fayda", "benefit", "benefits", "bienfait", "bienfaits", "فوائد", "ya3mel"]):
        return "product_benefits"

    if any(word in q for word in ["kifeh", "comment utiliser", "how should i use", "how to use", "كيف أستعمل", "nesta3ml", "usage", "قداش من شهر", "how long"]):
        return "product_usage"

    if product:
        return "product_info"
    
    product_request_words = [
        "produit",
        "products",
        "product",
        "yelzmni",
        "يلزمني",
        "nheb produit",
        "شنو يلزمني",
        "شنوة يلزمني",
        "شنوة المنتج",
        "شنو المنتج",
        "أي منتج",
        "behi",
        "best",
        "ahsen",
       "solution",
    ]

    weight_loss_words = [
        "naqs",
        "wazn",
        "ynaqsou",
        "yna9es",
        "n9as",
        "lose weight",
        "perdre du poids",
        "perte de poids",
        "نقص الوزن",
        "ننقص في الوزن",
        "إنقاص الوزن",
        "فقدان الوزن",
        "تنحيف",
    ]

    asks_for_product = any(
        word in q
        for word in product_request_words
    )

    mentions_weight_loss = any(
        word in q
        for word in weight_loss_words
    )

    if asks_for_product and mentions_weight_loss:
        return "product_recommendation"

    if mentions_weight_loss:
        return "weight_loss_advice"

    if asks_for_product:
         return "product_recommendation"

    if any(word in q for word in ["muscle", "prise de masse", "nzid", "wazn", "mass", "عضلات"]):
        return "muscle_gain_advice"

    # FIX: detect_intent n'avait aucun mot-cle sport/exercice -- une
    # question purement sportive ("quel sport je peux faire ?") tombait
    # sur "unknown" dans le champ intent renvoye, meme si la reponse
    # elle-meme etait bonne (car detect_intent_domain, une fonction
    # separee utilisee seulement pour adapter le ton, detectait bien
    # "exercise"). Les deux classifieurs sont maintenant coherents.
    if any(word in q for word in [
        "exercise", "sport", "training", "workout", "cardio",
        "walk", "walking", "run", "running", "fitness", "gym", "movement",
        "رياضة", "تمرين", "تمارين", "مشي", "لياقة",
    ]):
        return "exercise"

    if any(word in q for word in ["nekil", "eat", "manger", "meal", "repas", "ftor", "ghda", "3cha", "آكل"]):
        return "meal_suggestion"

    if any(word in q for word in ["calorie", "calories", "سعرات", "حريرات", "9addech", "qaddech"]):
        return "calorie_question"

    if any(word in q for word in [
        "produit", "products", "product", "les produit", "aandkom", "3andkom",
        "behi", "best", "ahsen", "haja", "solution"
    ]) and any(word in q for word in [
        "naqs", "wazn", "ynaqsou", "lose weight", "perdre du poids", "وزن",
        "constipation", "colon", "kebda", "kerch", "digest", "digestion",
        "bloating", "نفخة", "إمساك", "هضم", "كبد", "كرش", "belly"
    ]):
        return "product_recommendation"


    
    

    if any(word in q for word in ["shifa", "brand", "marque", "شنية شفاء", "ما هي شفاء"]):
        return "brand_info"

    return "unknown"


def build_intent_instruction(intent: str) -> str:
    instructions = {
        "greeting": "Reply briefly and warmly.",
        "weight_loss_advice": "Give practical weight-loss advice with 2 to 4 concrete suggestions.",
        "muscle_gain_advice": "Give practical muscle-gain advice with food and habit suggestions.",
        "exercise": "Suggest 2 to 4 concrete exercise or activity ideas adapted to the user's profile, age, and goal.",
        "meal_suggestion": "Suggest meal ideas adapted to the user's goal.",
        "calorie_question": "Answer simply about calories. If exact value is unknown, say it is an estimate.",
        "product_ingredients": (
            "List the product ingredients using only the product database. "
            "Do not invent missing quantities."
        ),
        "product_info": "Explain clearly what the product is, based only on the provided product information.",
        "product_benefits": "Explain product benefits based only on the provided product information.",
        "product_usage": "Explain briefly how to use the product in 1 to 2 sentences.",
        "product_recommendation": "Recommend the most relevant Shifa product naturally and explain briefly why it fits the user's goal.",
        "digestion_issue": "Give digestion advice and recommend Colon Detox only if relevant.",
        "stress_sleep_issue": "Give stress/sleep wellness advice and recommend Blood Detox only if relevant.",
        "brand_info": "Answer briefly about the brand using the brand information provided.",
        "delivery_info": "Answer only about delivery fees and delivery conditions using the provided offers. Do not invent delivery information.",
        "unknown": "Give a short, clear, safe, general answer without inventing facts.",
    }
    return instructions.get(intent, instructions["unknown"])


def detect_intent_domain(question: str) -> str:
    q = question.lower()

    if any(w in q for w in [
        "hi", "hello", "aslema", "slm", "salem", "bonjour", "مرحبا", "سلام"
    ]):
        return "greeting"

    if any(w in q for w in [
        # only explicit shifa/product family signals
        "shifa", "slim day", "slim night", "slim pack",
        "colon detox", "liver detox", "blood detox",
        "colon", "liver", "blood detox",
        "constipation", "digest", "digestion", "kerch", "bloating",
        "نفخة", "إمساك", "هضم", "كرش", "كبد",
        "فوائد منتج", "كيفاش نستعمل", "سعر منتج", "عرض شفاء"
    ]):
        return "shifa_products"

    if any(w in q for w in [
        "eat", "manger", "meal", "food", "calorie", "diet", "protein", "carb",
        "fat", "fiber", "vitamin", "fruit", "vegetable", "healthy food",
        "couscous", "pizza", "riz", "bread", "banana", "egg",
        "ناكل", "اكل", "وجبة", "وجبات", "سعرات", "تغذية", "شنوة ناكل",
        "regime", "régime", "nutrition", "light meals"
    ]):
        return "nutrition"

    if any(w in q for w in [
        "exercise", "sport", "training", "workout", "cardio", "walk", "walking",
        "run", "running", "fitness", "gym", "movement",
        "رياضة", "تمرين", "تمارين", "مشي", "لياقة"
    ]):
        return "exercise"

    if any(w in q for w in [
        "health", "wellness", "sleep", "healthy", "habit", "habits",
        "lifestyle", "energy", "stress", "digestion",
        "صحة", "نوم", "عادات", "رفاهة", "توتر", "طاقة"
    ]):
        return "health_wellness"

    return "unknown"

def is_short_followup(question: str) -> bool:
    q = question.lower().strip()
    short_followups = {
        "ey", "eya", "oui", "yes", "ok", "okay",
        "and then", "ensuite", "ba3d", "mba3d",
        "more", "akther", "زيد", "عادي", "donc", "alors"
    }
    return q in short_followups
def apply_weight_and_glycemic_priority(
    ranked_products: list[dict],
    signals: dict,
    merged_profile: dict,
    bmi: float | None,
) -> list[dict]:
    if not ranked_products:
        return ranked_products

    health = {
        str(item).lower().strip()
        for item in (
            signals.get("health_interests")
            or []
        )
    }

    recommended_signals = {
        str(item).lower().strip()
        for item in (
            signals.get(
                "recommended_product_signals"
            )
            or []
        )
    }

    goals = merged_profile.get("goals") or []

    if not isinstance(goals, list):
        goals = [goals]

    goals = {
        str(item).lower().strip()
        for item in goals
        if item
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
        all_signals & glycemic_signals
    )

    has_weight_goal = (
        bool(all_signals & weight_signals)
        or (
            bmi is not None
            and bmi >= 25
        )
    )

    if has_glycemic_need:
        preferred_name = (
            "Berberine & Ceylon Cinnamon"
        )

    elif has_weight_goal:
        preferred_name = "Slim Pack"

    else:
        return ranked_products

    preferred = [
        item
        for item in ranked_products
        if item.get("name") == preferred_name
    ]

    remaining = [
        item
        for item in ranked_products
        if item.get("name") != preferred_name
    ]

    return preferred + remaining
def recommend_shifa_products(
    question: str,
    merged_profile: dict | None = None,
    limit: int = 3,
) -> list[dict]:
    product_db = get_product_knowledge_dict()




    pseudo_signals = {
        "health_interests": [],
        "food_patterns": [],
        "recommended_product_signals": [],
        "detected_priority_need": None,
    }

    q = normalize_text(question)

    keyword_signal_map = {
        "respiratory_support": [
            "lung",
            "lungs",
            "breathing",
            "respiration",
            "poumon",
            "smoking",
            "tabac",
            "رئة",
            "تنفس",
            "تدخين",
        ],

        "glycemic_balance": [
            "blood sugar",
            "glucose",
            "glycémie",
            "insulin",
            "سكر الدم",
            "أنسولين",
        ],

        "constipation": [
            "constipation",
            "إمساك",
        ],

        "fiber_support": [
            "fiber",
            "fibre",
            "psyllium",
            "ألياف",
            "سيليوم",
        ],

        "bloating_gas": [
            "bloating",
            "gas",
            "gaz",
            "نفخة",
            "غازات",
        ],

        "detox": [
            "liver",
            "detox",
            "كبد",
            "سموم",
        ],

        "weight_loss": [
            "weight loss",
            "slimming",
            "perdre du poids",
            "نقص وزن",
            "تنحيف",
            "perte de poids",
            "maigrir",
            "نقص الوزن",
            "إنقاص الوزن",
            "فقدان الوزن",
            "ننقص في الوزن",
            "ننقص وزن",
            "نحب ننقص",
            "نحب نضعف",
            "تنحيف",
        ],

        "heart_support": [
            "heart",
            "circulation",
            "قلب",
            "دورة دموية",
        ],
    }

    for signal, keywords in keyword_signal_map.items():
        if any(keyword in q for keyword in keywords):
            pseudo_signals["health_interests"].append(
                signal
            )

    profile = merged_profile or {}

    bmi = None

    try:
        weight = profile.get("weight")
        height = profile.get("height")

        if weight and height:
            bmi = weight / ((height / 100) ** 2)
    except (TypeError, ZeroDivisionError):
        bmi = None

    ranked = score_products(
        product_db=product_db,
        signals=pseudo_signals,
        merged_profile=profile,
        bmi=bmi,
    )

    ranked = apply_digestive_product_priority(
        ranked,
        pseudo_signals,
    )

    ranked = apply_weight_and_glycemic_priority(
        ranked_products=ranked,
        signals=pseudo_signals,
        merged_profile=profile,
        bmi=bmi,
    )

    return ranked[:limit]

def is_out_of_scope(question: str, chat_history: list | None = None) -> bool:
    q = question.lower().strip()
    chat_history = chat_history or []

    short_followups = {
        "ey", "eyy", "oui", "yes", "ok", "okay", "d'accord", "dakord", "dacc",
        "aslema", "salem", "hello", "hi",
        "and then", "ensuite", "ba3d", "mba3d", "more", "akther", "زيد"
    }
    if q in short_followups:
        return False

    intent = detect_intent(question)
 

    user_language = detect_fallback_language(question)
    domain = detect_intent_domain(question)

    if intent in [
        "greeting",
        "price_offer_query",
        "weight_loss_advice",
        "muscle_gain_advice",
        "meal_suggestion",
        "calorie_question",
        "product_recommendation",
        "product_benefits",
        "product_usage",
        "product_info",
        "brand_info",
    ]:
        return False

    if domain in ["greeting", "shifa_products", "nutrition", "exercise", "health_wellness"]:
        return False

    last_product = get_last_product_from_history(chat_history)
    if last_product and len(q.split()) <= 4:
        return False

    return is_clearly_unrelated(question)
 
def is_clearly_unrelated(question: str) -> bool:
    q = question.lower().strip()

    unrelated_keywords = [
        "instagram caption", "caption", "bio instagram", "post caption",
        "who is", "qui est", "capital", "history",
        "actor", "actress", "singer", "movie", "football", "politics",
        "president", "programming", "math", "python code", "java code",
        "مغني", "ممثل", "سياسة", "رئيس", "عاصمة", "تاريخ", "كود", "معادلة",
        "شيرين", "سعد لمجرد"
    ]

    person_question_starters = [
        "chkoun", "chkun", "qui", "who is", "taarfou", "ta3ref", "تعرف", "شكون"
    ]

    in_scope_keywords = [
        "shifa", "slim", "colon", "liver", "blood detox",
        "nutrition", "diet", "meal", "food", "eat", "manger",
        "calorie", "exercise", "sport", "health", "wellness",
        "protein", "muscle", "sleep", "digestion", "healthy",
        "ناكل", "تغذية", "رياضة", "صحة", "سعرات", "هضم", "وزن", "كرش", "إمساك",
        "عضلات", "نوم", "طاقة", "وجبة", "وجبات"
    ]

    if any(q.startswith(x) for x in person_question_starters) and not any(k in q for k in in_scope_keywords):
        return True

    if any(k in q for k in unrelated_keywords):
        return True

    return False

def build_out_of_scope_answer(question: str) -> str:
    return (
        "نعتذر، أنا مساعد شفاء ومجالي يقتصر على منتجات شفاء، التغذية، السعرات، "
        "العادات الصحية والرفاهة. إذا تحب، اسألني مثلاً على فوائد منتج، طريقة استعماله، "
        "السعر، أو نصائح غذائية حسب هدفك."
    )

def is_price_or_offer_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "price", "prix", "thmen", "soum", "قداش", "سعر",
        "offer", "promo", "promotion", "عرض",
        "1", "2", "3", "piece", "pieces", "علبة", "علبتين", "ثلاثة"
    ]
    return any(k in q for k in keywords)


def format_quantity_offers(product_name: str | None, quantity_offers_db: dict) -> str:
    if not product_name or product_name not in quantity_offers_db:
        return ""

    offers = quantity_offers_db[product_name]

    formatted = ["Available quantity offers:"]

    for offer in offers:
        formatted.append(
            f"- {offer.get('title')}: "
            f"{offer.get('new_price')} {offer.get('currency')} "
            f"بدل {offer.get('old_price')} {offer.get('currency')} "
            f"({offer.get('discount_percent')}% تخفيض)، "
            f"{offer.get('delivery_text')}"
        )

    return "\n".join(formatted)

def get_last_intent_from_history(chat_history: list | None) -> str | None:
    if not chat_history:
        return None

    for msg in reversed(chat_history):
        content = msg.get("content", "")
        if "التوصيل" in content or "livraison" in content or "delivery" in content:
            return "delivery_info"

    return None

def get_best_quantity_offer(product_name: str | None, quantity_offers_db: dict) -> str:
    if not product_name or product_name not in quantity_offers_db:
        return ""

    offers = quantity_offers_db[product_name]
    best = max(offers, key=lambda x: x.get("discount_percent", 0))

    return (
        f"Best value offer: {best.get('title')} "
        f"بسعر {best.get('new_price')} {best.get('currency')} "
        f"بدل {best.get('old_price')} {best.get('currency')} "
        f"({best.get('discount_percent')}% discount), {best.get('delivery_text')}."
    )

def format_product_context(product_name: str | None, product_db: dict) -> str:
    product = get_product_safely(product_db, product_name)

    if not product:
        return ""

    p = product

    benefits = "\n".join([f"- {b}" for b in p.get("benefits", [])])
    ingredients = "\n".join([f"- {i}" for i in p.get("ingredients", [])])
    precautions = "\n".join([f"- {pr}" for pr in p.get("precautions", [])])

    return f"""
Product name: {p.get('name', '')}
Category: {p.get('category', '')}
Price: {p.get('price', '')} {p.get('currency', '')}
Old price: {p.get('old_price', '')}
Offer active: {p.get('offer_active', False)}
Offer title: {p.get('offer_title', '')}
Offer description: {p.get('offer_description', '')}
Pack size: {p.get('pack_size', '')}

Description:
{p.get('description', '')}

Benefits:
{benefits}

Ingredients:
{ingredients}

Usage:
{p.get('usage', '')}

Precautions:
{precautions}
"""

def format_chat_history(chat_history: list | None) -> str:
    if not chat_history:
        return "No previous conversation."

    formatted = []
    for msg in chat_history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
def get_product_safely(product_db: dict, product_name: str | None):
    if not product_name:
        return None

    normalized_name = product_name.strip().lower()

    for key, product in product_db.items():
        if str(key).strip().lower() == normalized_name:
            return product

        db_name = str(product.get("name", "")).strip().lower()
        if db_name == normalized_name:
            return product

    return None


def detect_fallback_language(
    text: str,
) -> str:
    text = normalize_text(text)

    # Arabic letters
    if any(
        "\u0600" <= char <= "\u06FF"
        for char in text
    ):
        return "ar"

    # Tunisian dialect written with Latin letters
    tunisian_markers = [
        "chnowa",
        "chneya",
        "chnouwa",
        "kifeh",
        "mta3",
        "mta",
        "nheb",
        "aandkom",
        "3andkom",
        "bqadeh",
        "bgideh",
        "yelzmni",
        "lel",
        "wazn",
        "na9es",
        "n9as",
        "behi",
        "faida",
        "faidet",
        "ya3mel",
        "nesta3mel",
        "nestaaml",
    ]

    if any(
        marker in text
        for marker in tunisian_markers
    ):
        return "ar"

    # French
    french_markers = [
        "produit",
        "produits",
        "bienfait",
        "bienfaits",
        "prix",
        "comment",
        "utiliser",
        "livraison",
        "minceur",
        "poumons",
        "pour",
        "avec",
        "quel",
        "quelle",
    ]

    if any(
        marker in text
        for marker in french_markers
    ):
        return "fr"

    return "en"

def find_matching_bundles(
    product_names: list[str],
    bundle_offers_db: list[dict],
) -> list[dict]:
    requested = {
        normalize_text(name)
        for name in product_names
        if name
    }

    if len(requested) < 2:
        return []

    matches = []

    for bundle in bundle_offers_db:
        bundle_products = {
            normalize_text(name)
            for name in (
                bundle.get("products")
                or []
            )
            if name
        }

        # The bundle must include all products requested by the user
        if requested.issubset(bundle_products):
            matches.append(bundle)

    return matches
def build_multi_product_price_answer(
    product_names: list[str],
    product_db: dict,
    bundle_offers_db: list[dict],
    language: str,
) -> tuple[str, list[dict], list[dict]]:
    price_items = []

    for product_name in product_names:
        product = get_product_safely(
            product_db,
            product_name,
        )

        if not product:
            continue

        price_items.append({
            "product": product_name,
            "price": product.get("price"),
            "old_price": product.get("old_price"),
            "currency": (
                product.get("currency")
                or "TND"
            ),
        })

    bundles = find_matching_bundles(
        product_names,
        bundle_offers_db,
    )

    language = (
        language or "ar"
    ).lower()

    lines = []

    if language.startswith("fr"):
        for item in price_items:
            name = item["product"]
            price = item["price"]
            old_price = item["old_price"]
            currency = item["currency"]

            if price is None:
                lines.append(
                    f"• Le prix de {name} n’est pas disponible actuellement."
                )
            elif old_price and old_price > price:
                lines.append(
                    f"• {name} : {price:g} {currency} "
                    f"au lieu de {old_price:g} {currency}."
                )
            else:
                lines.append(
                    f"• {name} : {price:g} {currency}."
                )

        if bundles:
            lines.append(
                "\nCes produits sont aussi disponibles ensemble dans une offre :"
            )

            for bundle in bundles:
                title = (
                    bundle.get("title")
                    or bundle.get("name")
                    or "Pack"
                )
                price = (
                    bundle.get("new_price")
                    or bundle.get("price")
                )
                old_price = bundle.get("old_price")
                currency = (
                    bundle.get("currency")
                    or "TND"
                )

                if price is None:
                    lines.append(
                        f"• {title} : prix non disponible."
                    )
                elif old_price and old_price > price:
                    lines.append(
                        f"• {title} : {price:g} {currency} "
                        f"au lieu de {old_price:g} {currency}."
                    )
                else:
                    lines.append(
                        f"• {title} : {price:g} {currency}."
                    )

    elif language.startswith("en"):
        for item in price_items:
            name = item["product"]
            price = item["price"]
            old_price = item["old_price"]
            currency = item["currency"]

            if price is None:
                lines.append(
                    f"• The price of {name} is currently unavailable."
                )
            elif old_price and old_price > price:
                lines.append(
                    f"• {name}: {price:g} {currency} "
                    f"instead of {old_price:g} {currency}."
                )
            else:
                lines.append(
                    f"• {name}: {price:g} {currency}."
                )

        if bundles:
            lines.append(
                "\nThese products are also available together in a bundle:"
            )

            for bundle in bundles:
                title = (
                    bundle.get("title")
                    or bundle.get("name")
                    or "Bundle"
                )
                price = (
                    bundle.get("new_price")
                    or bundle.get("price")
                )
                old_price = bundle.get("old_price")
                currency = (
                    bundle.get("currency")
                    or "TND"
                )

                if price is None:
                    lines.append(
                        f"• {title}: price unavailable."
                    )
                elif old_price and old_price > price:
                    lines.append(
                        f"• {title}: {price:g} {currency} "
                        f"instead of {old_price:g} {currency}."
                    )
                else:
                    lines.append(
                        f"• {title}: {price:g} {currency}."
                    )

    else:
        for item in price_items:
            name = item["product"]
            price = item["price"]
            old_price = item["old_price"]

            if price is None:
                lines.append(
                    f"• سعر {name} موش متوفر حالياً."
                )
            elif old_price and old_price > price:
                lines.append(
                    f"• {name}: {price:g} دينار "
                    f"عوض {old_price:g} دينار."
                )
            else:
                lines.append(
                    f"• {name}: {price:g} دينار."
                )

        if bundles:
            lines.append(
                "\nالمنتجات هاذم موجودين زادة مع بعضهم في عرض:"
            )

            for bundle in bundles:
                title = (
                    bundle.get("title")
                    or bundle.get("name")
                    or "باك"
                )
                price = (
                    bundle.get("new_price")
                    or bundle.get("price")
                )
                old_price = bundle.get("old_price")

                if price is None:
                    lines.append(
                        f"• {title}: السعر موش متوفر."
                    )
                elif old_price and old_price > price:
                    lines.append(
                        f"• {title}: {price:g} دينار "
                        f"عوض {old_price:g} دينار."
                    )
                else:
                    lines.append(
                        f"• {title}: {price:g} دينار."
                    )

    return (
        "\n".join(lines),
        price_items,
        bundles,
    )

def chatbot_response(question, user_profile=None, chat_history=None):
    chat_history = chat_history or []
    user_profile = user_profile or {}
    user_id = user_profile.get("user_id", "demo_user")
    stored_memory = get_user_memory(user_id)
    stored_profile = get_user_profile(user_id)
    product_db = get_product_knowledge_dict()
    quantity_offers_db = get_quantity_offers_dict()
    bundle_offers_db = get_bundle_offers()
    product_db = get_product_knowledge_dict()
   
  
    # FIX: merged_profile doit être construit AVANT d'être utilisé par
    # recommend_shifa_products ci-dessous (il était référencé avant sa
    # définition, ce qui provoquait un UnboundLocalError à chaque appel).
    merged_profile = {
        "user_id": user_id,
        "age": user_profile.get("age") or stored_profile.get("age"),
        "sex": user_profile.get("sex") or stored_profile.get("sex"),
        "weight": user_profile.get("weight") or stored_profile.get("weight"),
        "height": user_profile.get("height") or stored_profile.get("height"),
        "goals": user_profile.get("goals") or stored_profile.get("goals", []),
        "medical_conditions": user_profile.get("medical_conditions") or stored_profile.get("medical_conditions", []),
        "activity_info": user_profile.get("activity_info") or stored_memory.get("activity_info"),
    # Daily check-in memory
        "health_interests": stored_memory.get("health_interests", []),
        "recurring_food_patterns": stored_memory.get("recurring_food_patterns", []),
        "recurring_activity_patterns": stored_memory.get("recurring_activity_patterns", []),
        "last_meal_summary": stored_memory.get("last_meal_summary"),
        "last_activity_summary": stored_memory.get("last_activity_summary"),
        "last_detected_issue": stored_memory.get("last_detected_issue"),
        "consistency_score": stored_memory.get("consistency_score"),
        "language": (
            user_profile.get("language")
            or stored_profile.get("language")
            or detect_fallback_language(question)
        ),
    }

    ranked_products = recommend_shifa_products(
        question,
        merged_profile,
    )

    reason = None
    loyalty_recommendation = None
    if ranked_products:
        best = ranked_products[0]
        recommended_product = best["name"]

        reason = build_product_reason(
            product_name=best["name"],
            product=best["product"],
            matched_signals=best["matched_signals"],
            language="ar",
        )
    

    user_language = (
        user_profile.get("language")
        or stored_profile.get("language")
        or detect_fallback_language(question)
    )


    if is_out_of_scope(question, chat_history):
        answer = build_out_of_scope_answer(question)
        log_chat_interaction(user_id, {
            "question": question,
            "answer": answer,
            "intent": "out_of_scope",
            "detected_product": None,
            "recommended_product": None,
            "used_memory": bool(stored_memory),
        })
        return {
            "answer": answer,
            "intent": "out_of_scope",
            "detected_product": None,
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": "تنجم تسألني على منتجات شفاء، التغذية، السعرات، أو العادات الصحية.",
            "price_info": None,
            "offer_info": None
        }
    intent = detect_intent(question)
    user_language = detect_fallback_language(question)
    last_intent = get_last_intent_from_history(chat_history)

    if intent == "product_info" and last_intent == "delivery_info":
         intent = "delivery_info"
    profile_nutrition_rule = ""
    if intent in ["meal_suggestion", "weight_loss_advice", "calorie_question", "muscle_gain_advice"] or detect_intent_domain(question) in ["nutrition", "exercise", "health_wellness"]:
        profile_nutrition_rule = """
    For this question, you must adapt the answer to the user's profile.
    If the user asks about a food, evaluate it according to the user's goal.
    Example:
    - if the goal is weight loss, suggest moderation, lighter portions, and healthier combinations
    - if the goal is muscle gain, suggest enough protein and balanced carbs
    Do not answer in a generic way if profile information exists.
    """  
    
    category_matches = []

    detected_products = detect_products(
        question,
        product_db,
    )

    category_products, ambiguous_groups = (
        resolve_products_by_category(
            question,
            product_db,
        )
    )

# Add clear category-based products even when another
# exact product was already detected.
    detected_products = list(
        dict.fromkeys(
            detected_products
            + category_products
        )
    )

    category_matches = []

# Ask for clarification only when there is a genuinely
# ambiguous expression such as "detox".
    if ambiguous_groups:
        category_matches = list(
            dict.fromkeys(
                product_name
                for group in ambiguous_groups
                for product_name in group
            )
        )

# Fuzzy matching only when nothing was detected.
    if (
        not detected_products
        and not category_matches
    ):
        fuzzy_matches = (
            fuzzy_match_products_by_category(
                question,
                product_db,
            )
        )

        if len(fuzzy_matches) == 1:
            detected_products = fuzzy_matches

        elif len(fuzzy_matches) > 1:
            category_matches = fuzzy_matches

    detected_product = (
        detected_products[0]
        if len(detected_products) == 1
        else None
    )

# 3. Single resolved product
    

    print("========== PRODUCT DEBUG ==========")
    print("QUESTION:", question)
    print("INTENT:", intent)
    print("DETECTED PRODUCTS:", detected_products)
    print("CATEGORY PRODUCTS:", category_products)
    print("AMBIGUOUS GROUPS:", ambiguous_groups)
    print("CATEGORY MATCHES:", category_matches)
    print("===================================")

    
   
    if (
        len(category_matches) > 1
        and not detected_products
        and intent in [
            "price_offer_query",
            "product_ingredients",
            "product_benefits",
            "product_usage",
            "product_info",
            "delivery_info",
        ]
    ):
        answer = build_disambiguation_answer(
            question,
            category_matches,
        )

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": None,
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "loyalty_recommendation": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": None,
        }        

    last_product = get_last_product_from_history(chat_history)

    if not detected_product and intent == "price_offer_query":
        detected_product = last_product

    elif not detected_product and is_implicit_reference(question):
        detected_product = last_product

  

        log_chat_interaction(
            user_id,
            {
                "question": question,
                "answer": answer,
                "intent": "price_offer_query",
                "detected_product": ", ".join(
                    detected_products
                ),
                "recommended_product": None,
                "used_memory": False,
            },
        )

        return {
            "answer": answer,
            "intent": "price_offer_query",
            "detected_product": detected_products,
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "loyalty_recommendation": None,
            "follow_up_question": None,
            "price_info": product_prices,
            "offer_info": matching_bundles,
        }
    



        log_chat_interaction(
            user_id,
            {
                "question": question,
                "answer": answer,
                "intent": "price_offer_query",
                "detected_product": ", ".join(
                    detected_products
                ),
                "recommended_product": None,
                "used_memory": False,
            },
        )

        return {
            "answer": answer,
            "intent": "price_offer_query",
            "detected_product": detected_products,
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "loyalty_recommendation": None,
            "follow_up_question": None,
            "price_info": price_items,
            "offer_info": matching_bundles,
        }

    if intent == "price_offer_query" and detected_product:
        product = get_product_safely(product_db, detected_product)

        if not product:
            answer = f"ما لقيتش السعر الحالي متاع {detected_product}."
            price = None
        else:
            price = product.get("price")
            old_price = product.get("old_price")

            if price is None:
                answer = f"السعر الحالي متاع {detected_product} موش متوفر."
            elif old_price and old_price > price:
                answer = (
                    f"سعر {detected_product} هو {price:g} دينار "
                    f"عوض {old_price:g} دينار."
                )
            else:
                answer = f"سعر {detected_product} هو {price:g} دينار تونسي."

        log_chat_interaction(user_id, {
            "question": question,
            "answer": answer,
            "intent": "price_offer_query",
            "detected_product": detected_product,
            "recommended_product": None,
            "used_memory": bool(last_product),
        })

        return {
            "answer": answer,
            "intent": "price_offer_query",
            "detected_product": detected_product,
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": price,
            "offer_info": None,
        } 
    

# FIX: quand aucun produit exact n'a ete resolu (detected_product est
# None) mais que la question contient un terme de categorie generique
# ("produit detox", "produit minceur") qui correspond a PLUSIEURS
# produits, on demande une precision au lieu de laisser le LLM
# repondre dans le vide ("prix non specifie") alors que l'info existe
# bel et bien pour chacun des candidats.
    if not detected_product and intent in [
        "price_offer_query",
        "product_ingredients",
        "product_benefits",
        "product_usage",
        "product_info",
    ]:
        

        if len(category_matches) >= 2:
            answer = build_disambiguation_answer(question, category_matches)

            log_chat_interaction(user_id, {
                "question": question,
                "answer": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "used_memory": False,
            })

            return {
                "answer": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": build_disambiguation_answer(question, category_matches),
                "price_info": None,
                "offer_info": None,
            }

        if len(category_matches) == 1:
            detected_product = category_matches[0]

            # Le bloc price_offer_query original (plus haut) s'est deja
            # execute et a ete saute (detected_product etait None a ce
            # moment-la). On reproduit la meme reponse ici pour ce cas
            # a un seul candidat, sinon la question resterait sans prix.
            if intent == "price_offer_query":

                if len(detected_products) >= 2:
                    (
                        answer,
                        product_prices,
                        matching_bundles,
                    ) = format_multi_product_prices(
                        products=detected_products,
                        product_db=product_db,
                        bundle_offers_db=bundle_offers_db,
                        language=user_language,
                    )

                    log_chat_interaction(
                        user_id,
                        {
                            "question": question,
                            "answer": answer,
                            "intent": "price_offer_query",
                            "detected_product": ", ".join(
                                detected_products
                            ),
                            "recommended_product": None,
                            "used_memory": False,
                        },
                    )

                    return {
                        "answer": answer,
                        "response": answer,
                        "intent": "price_offer_query",
                        "detected_product": detected_products,
                        "recommended_product": None,
                        "recommendation_reason": None,
                        "meal_suggestion": None,
                        "calorie_info": None,
                        "usage_info": None,
                        "benefits_info": None,
                        "ingredients_info": None,
                        "precautions": None,
                        "lifestyle_suggestion": None,
                        "follow_up_question": None,
                        "price_info": product_prices,
                        "offer_info": matching_bundles,
                    }
        

                elif len(detected_products) == 1:
                    detected_product = detected_products[0]

                    product = get_product_safely(
                        product_db,
                        detected_product,
                    )

                    if not product or product.get("price") is None:
                        price = None
                        answer = (
                            f"السعر الحالي متاع {detected_product} "
                            f"موش متوفر."
                        )
                    else:
                        price = product.get("price")
                        old_price = product.get("old_price")

                        if old_price and old_price > price:
                            answer = (
                                f"سعر {detected_product} هو "
                                f"{price:g} دينار عوض "
                                f"{old_price:g} دينار."
                            )
                        else:
                            answer = (
                                f"سعر {detected_product} هو "
                                f"{price:g} دينار تونسي."
                            )

                    log_chat_interaction(user_id, {
                        "question": question,
                        "answer": answer,
                        "intent": "price_offer_query",
                        "detected_product": detected_product,
                        "recommended_product": None,
                        "used_memory": False,
                    })

                    return {
                        "answer": answer,
                        "intent": "price_offer_query",
                        "detected_product": detected_product,
                        "recommended_product": None,
                        "recommendation_reason": None,
                        "meal_suggestion": None,
                        "calorie_info": None,
                        "usage_info": None,
                        "benefits_info": None,
                        "precautions": None,
                        "lifestyle_suggestion": None,
                        "follow_up_question": None,
                        "price_info": price,
                        "offer_info": None,
                    }
                else:
        # no product detected
                    answer = build_disambiguation_answer(
                        question,
                        category_matches,
                    )

    if intent == "product_ingredients":
        products = get_products_context(
            detected_products,
            product_db,
        )

        if not products:
            if len(category_matches) > 1:
                answer = build_disambiguation_answer(
                    question,
                    category_matches,
                )
            else:
                answer = (
                    "على أنهي منتج تحب تعرف المكونات؟"
                )

            return {
                "answer": answer,
                "response": answer,
                "intent": intent,
                "detected_product": None, 
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "ingredients_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": answer,
                "price_info": None,
                "offer_info": None,
            }

        answer, ingredients_info = format_product_field_for_many(
            products=products,
            field="ingredients",
            title="مكونات",
            language=user_language,
        )

        product_names = [
            product.get("name")
            for product in products
        ] 

        log_chat_interaction(
            user_id,
            {
                "question": question,
                "answer": answer,
                "intent": intent,
                "detected_product": ", ".join(
                    product_names
                ),
                "recommended_product": None,
                "used_memory": bool(last_product),
            },
        )

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": (
                product_names[0]
                if len(product_names) == 1
                else product_names
            ),
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "ingredients_info": ingredients_info,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": None,
        }

    if intent == "product_benefits":
        products = get_products_context(
            detected_products,
            product_db,
        )

        if not products:
            answer = (
                build_disambiguation_answer(
                    question,
                    category_matches,
                )
                if len(category_matches) > 1
                else "على أنهي منتج تحب تعرف الفوائد؟"
            )
            log_chat_interaction(
                user_id,
                {
                    "question": question,
                    "answer": answer,
                    "intent": intent,
                    "detected_product": None,
                    "recommended_product": None,
                    "used_memory": bool(last_product),
                },
            )

            return {
                "answer": answer,
                "response": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "ingredients_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": answer,
                "price_info": None,
                "offer_info": None,
            }

        user_language = detect_fallback_language(
            question
        )

        titles = {
            "ar": "فوائد",
            "fr": "Bienfaits de",
            "en": "Benefits of",
        }

        answer, benefits_info = (
            format_product_field_for_many(
                products=products,
                field="benefits",
                title=titles.get(
                    user_language,
                    "Benefits of",
                ),
                language=user_language,
            )
        )


        print("========== BENEFITS DEBUG ==========")
        print("PRODUCTS CONTEXT:", products)
        print("GENERATED ANSWER:", answer)
        print("BENEFITS INFO:", benefits_info)
        print("====================================")

        product_names = [
            product.get("name")
            for product in products
        ]

        print("FINAL ANSWER RETURNED:", answer)

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": (
                product_names[0]
                if len(product_names) == 1
                else product_names
            ),
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": benefits_info,
            "ingredients_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": None,
        }

    if intent == "product_usage":
        products = get_products_context(
            detected_products,
            product_db,
        )

        if not products:
            answer = (
                build_disambiguation_answer(
                    question,
                    category_matches,
                )
                if len(category_matches) > 1
                else "على أنهي منتج تحب تعرف طريقة الاستعمال؟"
            )
            log_chat_interaction(
                user_id,
                {
                    "question": question,
                    "answer": answer,
                    "intent": intent,
                    "detected_product": None,
                    "recommended_product": None,
                    "used_memory": bool(last_product),
                },
            )

            return {
                "answer": answer,
                "response": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "ingredients_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": answer,
                "price_info": None,
                "offer_info": None,
            }

        answer, usage_info = format_product_field_for_many(
            products=products,
            field="usage",
            title="طريقة استعمال",
            language=user_language,
        )

        product_names = [
            product.get("name")
            for product in products
        ]

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": (
                product_names[0]
                if len(product_names) == 1
                else product_names
            ),
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": usage_info,
            "benefits_info": None,
            "ingredients_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": None,
        }



    if intent == "product_info":

        products = get_products_context(
            detected_products,
            product_db,
        )

        if not products:
 
            answer = (
                build_disambiguation_answer(
                    question,
                    category_matches,
                )
                if len(category_matches) > 1
                else "على أنهي منتج تحب معلومات؟"
            )

            return {
                "answer": answer,
                "response": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "ingredients_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": answer,
                "price_info": None,
                "offer_info": None,
            }

        answer, description_info = format_product_field_for_many(
            products=products,
            field="description",
            title="معلومات",
            language=user_language, 
             
        ) 

        product_names = [
            p["name"]
            for p in products
        ]

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": (
                product_names[0]
                if len(product_names) == 1
                else product_names
            ),
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "ingredients_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": None,
        }
    

    if intent == "delivery_info":

        if not detected_products:

            if len(category_matches) > 1:
                answer = build_disambiguation_answer(
                    question,
                    category_matches,
                )
            else:
                answer = (
                    "على أنهي منتج تحب تعرف معلومات التوصيل؟"
                )

            return {
                "answer": answer,
                "response": answer,
                "intent": intent,
                "detected_product": None,
                "recommended_product": None,
                "recommendation_reason": None,
                "meal_suggestion": None,
                "calorie_info": None,
                "usage_info": None,
                "benefits_info": None,
                "ingredients_info": None,
                "precautions": None,
                "lifestyle_suggestion": None,
                "follow_up_question": answer,
                "price_info": None,
                "offer_info": None,
            }

        delivery_info = (
            format_delivery_info_for_products(
                detected_products,
                quantity_offers_db,
                bundle_offers_db,
            )
        )

        answer = (
            "أكيد، هاني نعطيك معلومات التوصيل "
            "حسب المنتج والكمية:\n"
            f"{delivery_info}"
        )

        log_chat_interaction(
            user_id,
            {
                "question": question,
                "answer": answer,
                "intent": intent,
                "detected_product": ", ".join(
                    detected_products
                ),
                "recommended_product": None,
                "used_memory": bool(last_product),
            },
        )

        return {
            "answer": answer,
            "response": answer,
            "intent": intent,
            "detected_product": (
                detected_products[0]
                if len(detected_products) == 1
                else detected_products
            ),
            "recommended_product": None,
            "recommendation_reason": None,
            "meal_suggestion": None,
            "calorie_info": None,
            "usage_info": None,
            "benefits_info": None,
            "ingredients_info": None,
            "precautions": None,
            "lifestyle_suggestion": None,
            "follow_up_question": None,
            "price_info": None,
            "offer_info": delivery_info,
        }

   
    
    decision = build_decision_output(
        question=question,
        user_profile=merged_profile,
        intent=intent,
        detected_product=detected_product,
        product_db=product_db,
        quantity_offers=quantity_offers_db,
        bundle_offers=bundle_offers_db,
    )

    recommended_product = decision["recommended_product"]
    recommendation_reason = decision["recommendation_reason"]
    usage = decision["usage_info"]
    benefits = decision["benefits_info"]

    quantity_offers = format_quantity_offers(detected_product or recommended_product, quantity_offers_db)
    best_quantity_offer = get_best_quantity_offer(detected_product or recommended_product, quantity_offers_db)
    # RULES
    usage_rule = ""
    if intent == "product_usage":
        usage_rule = "Answer ONLY with usage instructions."

    price_rule = ""
    if intent in ["price_offer_query", "delivery_info"]:
        price_rule = """
    The user is asking about price, offer, or delivery.
    You must answer with available price/offers/delivery info from the provided product offers.
    If the question is only about delivery, focus only on delivery.
    Do not invent delivery fees.

    Keep the answer short, clear, and sales-oriented without sounding aggressive.
    """
    recommended_product_context = format_product_context(recommended_product, product_db)

    recommended_quantity_offers = format_quantity_offers(
        recommended_product,
        quantity_offers_db
    )

    recommended_best_offer = get_best_quantity_offer(
        recommended_product,
        quantity_offers_db
    )


    decision_context = f"""
    Decision layer output:
    - intent: {decision['intent']}
    - detected_product: {decision['detected_product']}
    - recommended_product: {decision['recommended_product']}
    - recommendation_reason: {decision['recommendation_reason']}
    - usage_info: {decision['usage_info']}
    - benefits_info: {decision['benefits_info']}
    - price_info: {decision['price_info']}
    - meal_suggestion: {decision['meal_suggestion']}
    - calorie_info: {decision['calorie_info']}
    - lifestyle_suggestion: {decision['lifestyle_suggestion']}
    - follow_up_question: {decision['follow_up_question']}
    """
    # PROMPT
    messages = [
        
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": TUNISIAN_EXAMPLES},
        {"role": "system", "content": f"Detected intent: {intent}"},
        
        {"role": "system", "content": f"Brand information:\n{BRAND_INFO}"},
        {"role": "system", "content": usage_rule},
        {"role": "system", "content": price_rule},
        {
            "role": "system",
            "content": """
        If the user asks about livraison/delivery/توصيل:
        - Use delivery_text or delivery_fee from quantity offers and bundle offers.
        - If 1 or 2 boxes have delivery fee, mention it.
        - If 3 boxes have free delivery, mention it.
        - Do not invent delivery information.
         """
        },
        {"role": "system", "content": f"Recommended best offer:\n{recommended_best_offer}"},
        
        {"role": "system", "content": f"Intent instruction: {build_intent_instruction(intent)}"},

        {"role": "system", "content": f"Best quantity offer:\n{best_quantity_offer}"},

        {"role": "system", "content": f"Product context:\n{format_product_context(detected_product, product_db)}"},
        {"role": "system", "content": f"Recommended product context:\n{format_product_context(recommended_product, product_db)}"},

        {"role": "system", "content": f"Offers:\n{quantity_offers}"},
        {"role": "system", "content": f"Recommended product offers:\n{format_quantity_offers(recommended_product, quantity_offers_db)}"},

        {"role": "system", "content": f"Recommendation reason: {recommendation_reason}"},

        {"role": "system", "content": f"User context:\n{build_user_context(merged_profile)}"},
        {"role": "system", "content": profile_nutrition_rule},

        {
            "role": "system",
            "content": """
    If the user asks for a product recommendation, you must recommend only products from Shifa.
    Do not mention generic supplement categories.
    If a relevant Shifa product exists, name it directly.
    For digestion, constipation, bloating, belly discomfort, or 'kerch', prefer Colon Detox.
    For liver or detox support, prefer Liver Detox.
    For weight loss, prefer Slim Pack.
    Never mention products, categories, or supplements that are not in the Shifa knowledge base.
    """
        },

        {
            "role": "system",
            "content": f"""
    Decision summary:
    - detected_product: {detected_product}
    - recommended_product: {recommended_product}
    - recommendation_reason: {recommendation_reason}
    """
        },

        {"role": "system", "content": decision_context},

        {"role": "system", "content": f"History:\n{format_chat_history(chat_history)}"},
        {"role": "user", "content": question}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content
    except Exception as exc:
        print("OpenAI call failed in chatbot_response:", exc)
        answer = FALLBACK_ANSWERS[detect_fallback_language(question)]

        log_chat_interaction(user_id, {
            "question": question,
            "answer": answer,
            "intent": "llm_error",
            "detected_product": decision.get("detected_product"),
            "recommended_product": decision.get("recommended_product"),
            "used_memory": bool(stored_memory),
        })

        return {
            "answer": answer,
            "intent": "llm_error",
            "detected_product": decision.get("detected_product"),
            "recommended_product": decision.get("recommended_product"),
            "recommendation_reason": decision.get("recommendation_reason"),
            "meal_suggestion": decision.get("meal_suggestion"),
            "calorie_info": decision.get("calorie_info"),
            "usage_info": decision.get("usage_info"),
            "benefits_info": decision.get("benefits_info"),
            "precautions": decision.get("precautions"),
            "lifestyle_suggestion": decision.get("lifestyle_suggestion"),
            "follow_up_question": None,
            "price_info": decision.get("price_info"),
            "offer_info": decision.get("offer_info"),
        }

    log_chat_interaction(user_id, {
        "question": question,
        "answer": answer,
        "intent": decision["intent"],
        "detected_product": decision["detected_product"],
        "recommended_product": decision["recommended_product"],
        "used_memory": bool(stored_memory),
    })
    memory_update = {
        "last_recommended_product": decision.get("recommended_product"),
    }

    health_interests = []
    notes = []
    q = question.lower()

    if any(w in q for w in ["constipation", "digest", "digestion", "colon", "bloating", "kerch", "نفخة", "إمساك", "هضم", "كرش"]):
        health_interests.append("digestion")
        notes.append("asked about digestion")
    if any(w in q for w in ["متقلق", "قلق", "stress", "anxiety", "مانجمش نرقد", "ما نرقدش", "نوم", "sleep"]):
        health_interests.append("stress_anxiety")
        notes.append("asked about stress or sleep")

    if any(w in q for w in ["liver", "detox", "kebda", "كبد", "سموم"]):
        health_interests.append("detox")
        notes.append("asked about liver or detox")

    if any(w in q for w in ["naqs", "lose weight", "perdre du poids", "وزن", "تنحيف", "slim"]):
        health_interests.append("weight_loss")
        notes.append("interested in weight loss")

    if any(w in q for w in ["muscle", "mass", "prise de masse", "عضلات"]):
        health_interests.append("muscle_gain")
        notes.append("interested in muscle gain")

    if health_interests:
        memory_update["health_interests"] = health_interests

    if decision.get("recommended_product"):
        memory_update["past_recommended_products"] = [decision["recommended_product"]]

    if notes:
        memory_update["notes"] = notes

    update_user_memory(user_id, memory_update)
    
    final_recommended_product = decision.get(
        "recommended_product"
    )

    loyalty_recommendation = None

    if final_recommended_product:
        user_language = (
            merged_profile.get("language")
            or user_profile.get("language")
            or detect_fallback_language(question)
        )

        loyalty_recommendation = (
            build_product_loyalty_message(
                user_code=user_id,
                product_name=final_recommended_product,
                language=user_language,
            )
        )

    return {
        "answer": answer,
        "intent": decision["intent"],
        "detected_product": decision["detected_product"],
        "recommended_product": decision["recommended_product"],
        "recommendation_reason": decision["recommendation_reason"],
        "meal_suggestion": decision["meal_suggestion"],
        "calorie_info": decision["calorie_info"],
        "usage_info": decision["usage_info"],
        "benefits_info": decision["benefits_info"],
        "precautions": decision["precautions"],
        "lifestyle_suggestion": decision["lifestyle_suggestion"],
        "loyalty_recommendation": loyalty_recommendation,
        "follow_up_question": decision["follow_up_question"],
        "price_info": decision["price_info"],
        "offer_info": decision["offer_info"],
    }