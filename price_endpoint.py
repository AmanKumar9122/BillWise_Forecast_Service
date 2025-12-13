"""
Price lookup endpoint for BillWise products.
Provides async price queries with TTL caching.
"""
from fastapi import HTTPException, Query
import os
import httpx
from cachetools import TTLCache, cached

BILLWISE_BASE = os.environ.get("BILLWISE_BASE", "http://localhost:8080")
BILLWISE_API_KEY = os.environ.get("BILLWISE_API_KEY", None)
PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL_SECONDS", "300"))

# Small in-memory TTL cache (productId -> unitPrice)
# maxsize=1000 products, ttl in seconds
price_cache = TTLCache(maxsize=1000, ttl=PRICE_CACHE_TTL)


async def _fetch_product(product_id: int):
    """
    Fetch product details from BillWise API.
    """
    headers = {}
    if BILLWISE_API_KEY:
        headers["Authorization"] = f"Bearer {BILLWISE_API_KEY}"
    
    url = f"{BILLWISE_BASE}/api/products/{product_id}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(url, headers=headers)
    
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch product from BillWise: {resp.status_code}"
        )
    return resp.json()


@cached(price_cache)
async def _get_unit_price_cached(product_id: int):
    """
    Get unit price for a product, cached with TTL.
    Cachetools decorator caches the result across calls.
    """
    j = await _fetch_product(product_id)
    
    # Try common price field names used in BillWise
    for key in ("sellingPricePerBaseUnit", "selling_price_per_base_unit", "sellingPrice", "price", "unitPrice"):
        if key in j and j[key] is not None:
            try:
                return float(j[key])
            except (ValueError, TypeError):
                continue
    
    raise HTTPException(
        status_code=404,
        detail="Unit price not found for product"
    )


async def get_price_async(product_id: int) -> dict:
    """
    Public async function to get unit price for a product.
    Returns dict with productId, unitPrice, source.
    Raises HTTPException on error.
    """
    try:
        unit_price = await _get_unit_price_cached(product_id)
        return {
            "productId": product_id,
            "unitPrice": unit_price,
            "source": "billwise"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
