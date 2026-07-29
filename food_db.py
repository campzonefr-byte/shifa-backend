"""
Base nutritionnelle multi-sources pour stabiliser l'estimation calorique
des check-ins en texte libre.

=======================================================================
SOURCES (à citer telles quelles dans le rapport de PFE)
=======================================================================

1) TCAT — Table de Composition des Aliments Tunisiens
   El Ati J., Béji C., Farhat A., Haddad S., Cherif S., Trabelsi T.,
   Danguir J., Gaigi S., Le Bihan G., Landais E., Eymard-Duvernay S.,
   Maire B., Delpeuch F. (2007). Table de composition des aliments
   tunisiens. Tunis : INNTA ; Montpellier : IRD, 294 p.
   ISBN 9789973998002.
   PDF officiel : https://horizon.documentation.ird.fr/exl-doc/pleins_textes/divers20-05/010041597.pdf
   -> Source PRIMAIRE et prioritaire : c'est la référence académique
      tunisienne officielle (228+ aliments/plats, 34 nutriments,
      noms en français/anglais/arabe littéraire/dialecte tunisien).
   -> Valeurs ci-dessous extraites directement du PDF (aliments crus,
      pour 100g de partie comestible). L'extraction automatique du
      document s'est arrêtée après une dizaine d'entrées (limite de
      l'outil, pas de la source) -- la table ci-dessous est donc un
      sous-ensemble fidèle mais partiel des 228+ aliments du document.
      Le reste peut être complété manuellement depuis le PDF (table
      des matières complète conservée en commentaire en bas de
      fichier pour référence).

2) CIQUAL — Table de composition nutritionnelle des aliments (ANSES,
   France). Agence nationale de sécurité sanitaire de l'alimentation,
   de l'environnement et du travail. Disponible sur https://ciqual.anses.fr
   -> Table française de référence (3480+ aliments), utilisée en
      COMPLÉMENT quand un aliment n'existe pas dans la TCAT (produits
      transformés, plats non spécifiquement tunisiens).
   -> Le site officiel ciqual.anses.fr ne permet pas l'extraction
      automatisée en masse (interface JS). Les valeurs ci-dessous
      proviennent de pages qui citent explicitement leur source
      Ciqual ; à re-vérifier sur ciqual.anses.fr avant publication
      académique si une précision au gramme près est nécessaire.

3) USDA FoodData Central. U.S. Department of Agriculture, Agricultural
   Research Service, Beltsville Human Nutrition Research Center.
   Disponible sur https://fdc.nal.usda.gov/ (données CC0, domaine public)
   -> Référence internationale, utilisée en dernier recours pour les
      aliments absents de TCAT et CIQUAL. FDC ID cité pour chaque
      entrée -> vérifiable directement sur fdc.nal.usda.gov/food-details/<FDC_ID>.

4) Dataset GitHub "Tunisian Food Database" (elyesmanai/Data-Science-Datasets)
   https://github.com/elyesmanai/Data-Science-Datasets/blob/main/EDA%20-%20Tunisian%20Food%20Database%20-%20Products.csv
   -> Produits DE MARQUE vendus en Tunisie (biscuits, chocolats, etc.),
      231 lignes, calories déjà annotées par marque/produit.
   -> Fichier protégé par robots.txt : non téléchargeable par un outil
      automatisé. Les quelques entrées ci-dessous viennent d'un extrait
      visible publiquement (aperçu de recherche) ; pour le jeu complet,
      téléchargez-le manuellement depuis GitHub ("Download raw file")
      et exécutez load_github_branded_csv() ci-dessous dessus.

=======================================================================
PRIORITÉ DE RECHERCHE (dans match_food_item)
=======================================================================
1. Plats tunisiens composites (COMPOSITE_DISHES) -- reconnaissance
   rapide de ce que les gens tapent réellement dans un check-in
   ("couscous poulet", "mloukhia"...). ATTENTION : ce ne sont PAS des
   valeurs officielles TCAT/CIQUAL/USDA (ces sources ne couvrent que
   des ingrédients crus, pas des plats composés) -- ce sont des
   estimations internes, à valider avec un(e) nutritionniste avant
   production. Champ "source": "internal_estimate" explicite.
2. Ingrédients bruts officiels (TCAT en priorité, puis CIQUAL, puis USDA)
3. Produits de marque tunisiens (GitHub dataset)
4. Fallback LLM si rien n'est trouvé
"""

from difflib import SequenceMatcher

# =======================================================================
# 1) TCAT -- valeurs officielles extraites du PDF (aliments crus, /100g)
# =======================================================================
TCAT_INGREDIENTS = {
    "amidon": {
        "aliases": ["amidon", "cornstarch", "maizena"],
        "kcal_100g": 381, "protein_g": 0.26, "fat_g": 0.05, "carbs_g": 91.27,
        "source": "TCAT", "tcat_page": 4,
    },
    "biscotte_ordinaire": {
        "aliases": ["biscotte", "biscotte ordinaire", "toast craker salted"],
        "kcal_100g": 390, "protein_g": 12.10, "fat_g": 3.20, "carbs_g": 76.60,
        "source": "TCAT", "tcat_page": 5,
    },
    "biscotte_sans_sel": {
        "aliases": ["biscotte sans sel", "toast craker unsalted"],
        "kcal_100g": 390, "protein_g": 12.10, "fat_g": 3.20, "carbs_g": 76.60,
        "source": "TCAT", "tcat_page": 6,
    },
    "ble_graines": {
        "aliases": ["ble", "ble en graines", "wheat dry", "قمح"],
        "kcal_100g": 327, "protein_g": 12.61, "fat_g": 1.54, "carbs_g": 65.68,
        "source": "TCAT", "tcat_page": 7,
    },
    "brioche": {
        "aliases": ["brioche"],
        "kcal_100g": 406, "protein_g": 8.20, "fat_g": 21.00, "carbs_g": 45.80,
        "source": "TCAT", "tcat_page": 8,
    },
    "couscous_sec": {
        "aliases": ["couscous sec", "couscous cru", "couscous-dry-cup", "كسكسي"],
        "kcal_100g": 350, "protein_g": 13.44, "fat_g": 1.50, "carbs_g": 73.98,
        "source": "TCAT", "tcat_page": 9,
    },
    "farine_blanche": {
        "aliases": ["farine", "farine blanche", "wheat flour white", "طحين"],
        "kcal_100g": 364, "protein_g": 11.00, "fat_g": 1.20, "carbs_g": 74.70,
        "source": "TCAT", "tcat_page": 10,
    },
    "farine_orge": {
        "aliases": ["farine d'orge", "barley flour"],
        "kcal_100g": 380, "protein_g": 12.00, "fat_g": 4.00, "carbs_g": 76.00,
        "source": "TCAT", "tcat_page": 11,
    },
    "mais_graines_seches": {
        "aliases": ["mais", "mais graines seches", "corn dry seeds", "ذرة"],
        "kcal_100g": 365, "protein_g": 9.42, "fat_g": 4.74, "carbs_g": 74.26,
        "source": "TCAT", "tcat_page": 12,
    },
    "orge_perle_cru": {
        "aliases": ["orge", "orge perle", "pearled barley", "شعير"],
        "kcal_100g": 352, "protein_g": 9.91, "fat_g": 1.16, "carbs_g": 77.72,
        "source": "TCAT", "tcat_page": 13,
    },
}

# Index complet des 228+ aliments/plats couverts par la TCAT (noms
# français/anglais tels qu'en table des matières du PDF) -- utile pour
# savoir QUOI compléter en priorité, même si les valeurs numériques ne
# sont pas encore toutes saisies ici. Voir le PDF officiel pour les
# valeurs manquantes (pages 14-272).
TCAT_KNOWN_FOOD_NAMES = [
    "Pain au chocolat", "Pain de mie", "Pain, baguette", "Pain, complet",
    "Pain, mbassess", "Pain, mlawi", "Pain, orge", "Pain, tabouna",
    "Pain, tagine", "Pâte alimentaire crue", "Pâte feuilletée crue",
    "Riz cru", "Semoule", "Sorgho", "Patate douce crue",
    "Pomme de terre frite", "Pomme de terre crue", "Fenugrec",
    "Fève sèche crue", "Haricot blanc sec cru", "Lentille sèche",
    "Pois chiches secs", "Artichaut cru", "Aubergine crue", "Bette crue",
    "Betterave rouge crue", "Carotte crue", "Céleri", "Champignon cru",
    "Chou vert cru", "Chou-fleur cru", "Concombre cru", "Courge crue",
    "Courgette crue", "Épinard cru", "Fenouil", "Fève fraîche crue",
    "Gombo cru", "Haricot vert cru", "Laitue crue", "Navet cru",
    "Oignon cru", "Persil frais", "Petit pois cru", "Poireau cru",
    "Poivron vert cru", "Radis cru", "Tomate concentrée", "Tomate crue",
    "Abricot", "Ananas frais", "Avocat frais", "Banane pulpe fraîche",
    "Datte pulpe et peau", "Citron pulpe frais", "Clémentine/mandarine",
    "Figue de barbarie", "Figue sèche", "Melon pulpe frais",
    "Fraise fraîche", "Grenade fraîche", "Jus d'orange frais",
    "Mangue pulpe fraîche", "Olive noire en saumure",
    "Olive verte en saumure", "Orange douce pulpe fraîche",
    "Pastèque pulpe fraîche", "Pêche", "Pomme fraîche", "Raisin frais",
    "Amande sèche", "Cacahuète grillée", "Noisette naturelle",
    "Noix naturelle", "Pistache grillée", "Sésame graine",
    "Foie d'agneau cru", "Rognon d'agneau cru", "Viande de lapin crue",
    "Viande de bœuf crue", "Viande de chameau crue", "Viande de chèvre crue",
    "Viande d'agneau crue", "Viande de poulet crue", "Viande de dinde crue",
    "Anchois frais", "Calamar cru", "Thon conserve", "Maquereau cru",
    "Merlu cru", "Moule crue", "Poulpe cru", "Sardine conserve",
    "Sardine crue", "Seiche crue", "Sole crue", "Crevette crue",
    "Œuf frais", "Lait entier UHT", "Leben", "Rayeb",
    "Yaourt au lait demi-écrémé nature", "Fromage pâte dure",
    "Beurre", "Huile d'olive extra vierge", "Huile de tournesol",
    "Margarine", "Smen", "Chamia", "Confiture", "Glace", "Miel",
    "Sucre", "Ail frais", "Cannelle", "Cumin en poudre", "Harissa",
    "Menthe fraîche", "Piment rouge moulu", "Café noir", "Thé",
    # ... liste complète des 228+ items dans le PDF officiel (voir
    # sommaire pages i-iv du document source)
]



# =======================================================================
# 2) CIQUAL (ANSES) -- valeurs complémentaires, citées via sources
#    secondaires qui référencent explicitement Ciqual. À reconfirmer
#    sur ciqual.anses.fr si une précision académique stricte est requise.
# =======================================================================
CIQUAL_INGREDIENTS = {
    "semoule_couscous_cru": {
        "aliases": ["semoule crue", "couscous semoule crue"],
        "kcal_100g": 376, "protein_g": 13.0, "fat_g": 0.6, "carbs_g": 77.0,
        "source": "CIQUAL", "note": "ANSES/Ciqual 2024, via sorn.fr (à revérifier sur ciqual.anses.fr)",
    },
    "legumes_couscous_cuits": {
        "aliases": ["legumes couscous cuits", "legumes pour couscous"],
        "kcal_100g": 20, "protein_g": None, "fat_g": None, "carbs_g": None,
        "source": "CIQUAL", "note": "Ciqual 2017, via alimentation-et-nutrition.fr",
    },
}

# =======================================================================
# 3) USDA FoodData Central -- FDC ID cité pour vérification directe
# =======================================================================
USDA_INGREDIENTS = {
    "poulet_blanc_cru": {
        "aliases": ["poulet cru", "blanc de poulet cru", "chicken breast raw", "دجاج نيء"],
        "kcal_100g": 114, "protein_g": 21.2, "fat_g": 2.6, "carbs_g": 0.0,
        "source": "USDA", "fdc_id": 171077,
        "fdc_url": "https://fdc.nal.usda.gov/food-details/171077/nutrients",
    },
    "poulet_blanc_cuit": {
        "aliases": ["poulet cuit", "blanc de poulet cuit", "chicken breast cooked", "poulet grille cuit"],
        "kcal_100g": 165, "protein_g": 31.0, "fat_g": 3.6, "carbs_g": 0.0,
        "source": "USDA", "fdc_id": 171477,
        "fdc_url": "https://fdc.nal.usda.gov/food-details/171477/nutrients",
    },
}

# =======================================================================
# 4) Produits de marque tunisiens -- échantillon du dataset GitHub
#    (extrait visible publiquement ; jeu complet à récupérer
#    manuellement, voir load_github_branded_csv() plus bas)
# =======================================================================
GITHUB_BRANDED_PRODUCTS = {
    "saida_break_kids_chocolat": {
        "aliases": ["break kids chocolat", "saida break kids"],
        "kcal_100g": 340, "protein_g": 4.5, "fat_g": 12.5, "carbs_g": 52.6,
        "source": "GitHub:Tunisian Food Database (elyesmanai)", "producer": "Saida",
    },
    "saida_plum_cakes": {
        "aliases": ["plum cakes", "plum-cakes", "saida plum cake"],
        "kcal_100g": 359, "protein_g": 5.58, "fat_g": 17.74, "carbs_g": 44.30,
        "source": "GitHub:Tunisian Food Database (elyesmanai)", "producer": "Saida",
    },
    "saida_falfoul_nounours": {
        "aliases": ["falfoul nounours", "chocolat falfoul"],
        "kcal_100g": 364, "protein_g": 0.37, "fat_g": 0.3, "carbs_g": 90.9,
        "source": "GitHub:Tunisian Food Database (elyesmanai)", "producer": "Saida",
    },
}


def load_github_branded_csv(csv_path: str) -> dict:
    """À utiliser une fois le CSV téléchargé manuellement depuis GitHub
    (le fichier est protégé par robots.txt, non récupérable par un outil
    automatisé). Colonnes attendues : Producer,Category,Product,
    Calories (100g),Lipides,Satures,Glucides,Sucre,Protein,Sel,Portion Size.

    Exemple :
        products = load_github_branded_csv("tunisian_food_products.csv")
        GITHUB_BRANDED_PRODUCTS.update(products)
    """
    import csv as csv_module
    import re

    def parse_number(value: str):
        if not value:
            return None
        cleaned = re.sub(r"[^\d.,]", "", value).replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    products = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            name = (row.get("Product") or "").strip()
            if not name:
                continue
            key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            products[key] = {
                "aliases": [name.lower()],
                "kcal_100g": parse_number(row.get("Calories (100g)")),
                "protein_g": parse_number(row.get("Protein")),
                "fat_g": parse_number(row.get("Lipides (Fat)")),
                "carbs_g": parse_number(row.get("Glucides (Carbs)")),
                "source": "GitHub:Tunisian Food Database (elyesmanai)",
                "producer": row.get("Producer"),
            }
    return products


# =======================================================================
# 5) Plats tunisiens composites -- ESTIMATIONS INTERNES (pas de source
#    officielle pour un plat composé -- TCAT/CIQUAL/USDA ne couvrent que
#    des ingrédients bruts). À faire valider par un(e) nutritionniste.
# =======================================================================
COMPOSITE_DISHES = {
    "couscous_poulet": {
        "aliases": ["couscous poulet", "couscous dajaj", "كسكسي بالدجاج", "كسكسي دجاج"],
        "kcal_small": 450, "kcal_medium": 650, "kcal_large": 900,
        "source": "internal_estimate",
    },
    "couscous_legumes": {
        "aliases": ["couscous khodhra", "couscous vegetarien", "كسكسي بالخضرة"],
        "kcal_small": 350, "kcal_medium": 500, "kcal_large": 700,
        "source": "internal_estimate",
    },
    "mloukhia": {
        "aliases": ["mloukhia", "mloukheya", "ملوخية"],
        "kcal_small": 300, "kcal_medium": 450, "kcal_large": 600,
        "source": "internal_estimate",
    },
    "brik": {
        "aliases": ["brik", "brik a l'oeuf", "بريك"],
        "kcal_small": 200, "kcal_medium": 280, "kcal_large": 350,
        "source": "internal_estimate",
    },
    "makloub": {
        "aliases": ["makloub", "مقلوب"],
        "kcal_small": 350, "kcal_medium": 500, "kcal_large": 650,
        "source": "internal_estimate",
    },
    "chorba": {
        "aliases": ["chorba", "soupe", "شربة"],
        "kcal_small": 150, "kcal_medium": 220, "kcal_large": 300,
        "source": "internal_estimate",
    },
    "tajine_tunisien": {
        "aliases": ["tajine", "tajine tounsi", "طاجين"],
        "kcal_small": 250, "kcal_medium": 380, "kcal_large": 500,
        "source": "internal_estimate",
    },
    "pates_sauce": {
        "aliases": ["pate", "pates", "maqarouna", "ماكرونة", "spaghetti"],
        "kcal_small": 350, "kcal_medium": 550, "kcal_large": 750,
        "source": "internal_estimate",
    },
    "salade_mechouia": {
        "aliases": ["mechouia", "salade mechouia", "مشوية"],
        "kcal_small": 100, "kcal_medium": 150, "kcal_large": 200,
        "source": "internal_estimate",
    },
    "salade_verte": {
        "aliases": ["salade", "salade verte", "سلطة"],
        "kcal_small": 60, "kcal_medium": 100, "kcal_large": 150,
        "source": "internal_estimate",
    },
    "poisson_grille": {
        "aliases": ["poisson", "hout", "حوت", "poisson grille"],
        "kcal_small": 180, "kcal_medium": 280, "kcal_large": 380,
        "source": "internal_estimate",
    },
    "oeufs": {
        "aliases": ["oeuf", "oeufs", "bayd", "بيض"],
        "kcal_small": 80, "kcal_medium": 160, "kcal_large": 240,
        "source": "internal_estimate",
    },
    "pain": {
        "aliases": ["pain", "khobz", "خبز"],
        "kcal_small": 120, "kcal_medium": 200, "kcal_large": 300,
        "source": "internal_estimate",
    },
    "makroudh": {
        "aliases": ["makroudh", "مقروض"],
        "kcal_small": 150, "kcal_medium": 250, "kcal_large": 350,
        "source": "internal_estimate",
    },
    "fruits": {
        "aliases": ["fruit", "fruits", "ghalla", "غلة"],
        "kcal_small": 60, "kcal_medium": 100, "kcal_large": 150,
        "source": "internal_estimate",
    },
}

_PORTION_ALIASES = {
    "small": ["small", "petit", "petite", "sghir", "صغير"],
    "medium": ["medium", "moyen", "moyenne", "wast", "وسط"],
    "large": ["large", "grand", "grande", "kbir", "كبير"],
}

# Facteur approximatif pour convertir un aliment "brut /100g" en 3 tailles
# de portion (petite ~70g, moyenne ~150g, grande ~250g) quand on matche
# une entrée TCAT/CIQUAL/USDA au lieu d'un plat composite.
_PORTION_GRAMS = {"small": 70, "medium": 150, "large": 250}


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _best_match_in_table(normalized: str, table: dict, threshold: float):
    best_key, best_entry, best_score = None, None, 0.0
    for key, entry in table.items():
        candidates = [key.replace("_", " ")] + entry.get("aliases", [])
        for candidate in candidates:
            score = _similarity(normalized, _normalize(candidate))
            if _normalize(candidate) in normalized or normalized in _normalize(candidate):
                score = max(score, 0.9)
            if score > best_score:
                best_key, best_entry, best_score = key, entry, score
    if best_score >= threshold:
        return best_key, best_entry
    return None, None


def normalize_portion(portion_text: str) -> str:
    normalized = _normalize(portion_text)
    for size, aliases in _PORTION_ALIASES.items():
        if any(a in normalized for a in aliases):
            return size
    return "medium"


def match_food_item(item_name: str, threshold: float = 0.75) -> dict:
    """Cherche l'aliment dans l'ordre de priorité :
    1. plats composites tunisiens (estimation interne)
    2. TCAT (officiel)
    3. CIQUAL (officiel, complément)
    4. USDA (officiel, complément)
    5. produits de marque (GitHub dataset)
    Retourne un dict avec la source utilisée, ou source=None si rien trouvé.
    """
    normalized = _normalize(item_name)
    if not normalized:
        return {"matched": False, "source": None}

    key, entry = _best_match_in_table(normalized, COMPOSITE_DISHES, threshold)
    if entry:
        return {"matched": True, "key": key, "entry": entry, "table": "composite_dish"}

    for table, table_name in [
        (TCAT_INGREDIENTS, "tcat"),
        (CIQUAL_INGREDIENTS, "ciqual"),
        (USDA_INGREDIENTS, "usda"),
        (GITHUB_BRANDED_PRODUCTS, "github_branded"),
    ]:
        key, entry = _best_match_in_table(normalized, table, threshold)
        if entry:
            return {"matched": True, "key": key, "entry": entry, "table": table_name}

    return {"matched": False, "source": None}


def estimate_calories_for_item(item_name: str, portion_text: str = "medium") -> dict:
    portion = normalize_portion(portion_text)
    match = match_food_item(item_name)

    if not match["matched"]:
        return {
            "item": item_name,
            "matched_food": None,
            "portion": portion,
            "estimated_calories": None,
            "source": "llm_fallback_needed",
        }

    entry = match["entry"]

    if match["table"] == "composite_dish":
        calories = entry[f"kcal_{portion}"]
    else:
        # ingrédient brut /100g -> on applique un grammage de portion
        kcal_per_100g = entry.get("kcal_100g")
        calories = (
            round(kcal_per_100g * _PORTION_GRAMS[portion] / 100)
            if kcal_per_100g is not None else None
        )

    return {
        "item": item_name,
        "matched_food": match["key"],
        "portion": portion,
        "estimated_calories": calories,
        "source": entry.get("source", match["table"]),
    }


def estimate_meal_calories(food_items: list[dict]) -> dict:
    """food_items: [{"name": "couscous poulet", "portion": "medium"}, ...]"""
    results = [
        estimate_calories_for_item(item.get("name", ""), item.get("portion", "medium"))
        for item in food_items
    ]

    known_total = sum(r["estimated_calories"] for r in results if r["estimated_calories"] is not None)
    unknown_items = [r["item"] for r in results if r["estimated_calories"] is None]

    return {
        "items": results,
        "known_calories_total": known_total,
        "items_needing_llm_estimate": unknown_items,
        "confidence": "high" if not unknown_items else "medium" if known_total > 0 else "low",
    }