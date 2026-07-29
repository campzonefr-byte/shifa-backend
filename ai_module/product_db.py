from ai_module.supabase_client import supabase


def upsert_product(product: dict):
    row = {
        "name": product.get("name"),
        "category": product.get("category"),
        "price": product.get("price"),
        "currency": product.get("currency", "TND"),
        "old_price": product.get("old_price"),
        "pack_size": product.get("pack_size"),
        "description": product.get("description"),
        "usage": product.get("usage"),
        "benefits": product.get("benefits", []),
        "ingredients": product.get("ingredients", []),
        "precautions": product.get("precautions", []),
        "quantity_offers": product.get("quantity_offers", []),
        "tags": product.get("tags", []),
        "image_emoji": product.get("image_emoji", "🌿"),
        "in_stock": product.get("in_stock", True),
    }

    return (
        supabase
        .table("products")
        .upsert(row, on_conflict="name")
        .execute()
    )



def get_all_products():
    res = supabase.table("products").select("*").execute()
    return res.data or []
def get_quantity_offers_dict() -> dict:
    products = get_all_products()

    return {
        product["name"]: product.get("quantity_offers", [])
        for product in products
        if product.get("name")
        and product.get("quantity_offers")
    }
def get_bundle_offers():
    return supabase.table("bundle_offers").select("*").execute().data
def get_product_by_name(product_name: str | None):
    if not product_name:
        return None

    res = (
        supabase.table("products")
        .select("*")
        .eq("name", product_name)
        .execute()
    )

    return res.data[0] if res.data else None

def get_product_by_name(product_name: str | None):
    if not product_name:
        return None

    clean_name = product_name.strip()

    res = (
        supabase.table("products")
        .select("*")
        .ilike("name", clean_name)
        .limit(1)
        .execute()
    )

    return res.data[0] if res.data else None


def get_product_knowledge_dict():
    products = get_all_products()
    return {p["name"]: p for p in products}


def get_quantity_offers_dict():
    products = get_all_products()
    return {
        p["name"]: p.get("quantity_offers") or []
        for p in products
        if p.get("quantity_offers")
    }


def get_bundle_offers():
    res = (
        supabase.table("bundle_offers")
        .select("*")
        .eq("active", True)
        .execute()
    )
    return res.data or []

def upsert_bundle_offer(bundle: dict):
    row = {
        "title": bundle.get("title"),
        "products": bundle.get("products", []),
        "old_price": bundle.get("old_price"),
        "new_price": bundle.get("new_price"),
        "currency": bundle.get("currency"),
        "delivery_fee": bundle.get("delivery_fee"),
        "description": bundle.get("description"),
        "active": bundle.get("offer_active", True),
    }
    print(row)

    return supabase.table("bundle_offers").insert(row).execute()