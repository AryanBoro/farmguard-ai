"""
FarmGuard AI — Weather Service
Fetches real weather data and computes disease risk based on conditions.
"""

import os
import httpx
from typing import Optional

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_BASE_URL = "https://api.weatherapi.com/v1/current.json"


async def get_weather(location: str) -> Optional[dict]:
    """
    Fetch current weather for a location via WeatherAPI.com.
    Returns None if API key not set or request fails.
    """
    if not WEATHER_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                WEATHER_BASE_URL,
                params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"}
            )
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        return {
            "location": f"{data['location']['name']}, {data['location']['country']}",
            "temp_c": current["temp_c"],
            "humidity": current["humidity"],
            "precip_mm": current["precip_mm"],
            "condition": current["condition"]["text"],
            "wind_kph": current["wind_kph"],
            "cloud": current["cloud"],
            "is_day": bool(current["is_day"])
        }
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return None


def compute_disease_risk(weather: Optional[dict], disease_weather_risk: dict) -> dict:
    """
    Assess environmental risk for a given disease based on current weather.

    Args:
        weather: dict from get_weather(), or None
        disease_weather_risk: weather_risk dict from REMEDY_DB entry

    Returns:
        risk_assessment dict with level, score, and contributing factors
    """
    if weather is None:
        return {
            "level": "unknown",
            "score": None,
            "message": "Weather data unavailable. Provide a WEATHER_API_KEY for risk assessment.",
            "factors": []
        }

    score = 0
    factors = []

    humidity = weather["humidity"]
    temp = weather["temp_c"]
    rain = weather["precip_mm"]

    # High humidity risk
    if disease_weather_risk.get("high_humidity") and humidity >= 80:
        score += 30
        factors.append(f"High humidity ({humidity}%) favors pathogen spread")
    elif disease_weather_risk.get("high_humidity") and humidity >= 65:
        score += 15
        factors.append(f"Moderate humidity ({humidity}%) — monitor conditions")

    # Rain-triggered risk
    if disease_weather_risk.get("rain_triggered") and rain > 0:
        score += 25
        factors.append(f"Active precipitation ({rain}mm) — ideal infection conditions")
    elif disease_weather_risk.get("rain_triggered") and weather["cloud"] > 70:
        score += 10
        factors.append("Overcast conditions with potential for rain")

    # Temperature range check
    temp_range = disease_weather_risk.get("temp_range", "any")
    if temp_range != "any":
        in_range = _temp_in_range(temp, temp_range)
        if in_range:
            score += 25
            factors.append(f"Temperature ({temp}°C) is within optimal disease range ({temp_range})")
        elif abs(_temp_distance(temp, temp_range)) < 5:
            score += 10
            factors.append(f"Temperature ({temp}°C) is near disease risk range ({temp_range})")

    # Hot dry = spider mite risk (special case)
    if temp > 30 and humidity < 50:
        score += 15
        factors.append(f"Hot dry conditions ({temp}°C, {humidity}% RH) favor spider mites")

    # Cap at 100
    score = min(score, 100)

    if score >= 70:
        level = "high"
        message = "⚠️ Current weather conditions are highly favorable for disease development. Take immediate preventive action."
    elif score >= 40:
        level = "moderate"
        message = "🟡 Weather conditions present moderate disease risk. Increase scouting frequency."
    elif score >= 15:
        level = "low"
        message = "🟢 Low risk from weather. Continue regular monitoring."
    else:
        level = "minimal"
        message = "✅ Current conditions are not favorable for this disease."

    return {
        "level": level,
        "score": score,
        "message": message,
        "factors": factors,
        "weather_snapshot": {
            "temp_c": temp,
            "humidity": humidity,
            "precip_mm": rain,
            "condition": weather["condition"]
        }
    }


def _temp_in_range(temp: float, temp_range: str) -> bool:
    """Check if temp falls within a range string like '13-24°C' or '>30°C'."""
    try:
        if temp_range.startswith(">"):
            threshold = float(temp_range[1:].replace("°C", ""))
            return temp > threshold
        elif temp_range.startswith("<"):
            threshold = float(temp_range[1:].replace("°C", ""))
            return temp < threshold
        elif "-" in temp_range:
            low, high = temp_range.replace("°C", "").split("-")
            return float(low) <= temp <= float(high)
    except Exception:
        pass
    return False


def _temp_distance(temp: float, temp_range: str) -> float:
    """Return how far temp is from the range (negative = below, positive = above)."""
    try:
        if "-" in temp_range:
            low, high = temp_range.replace("°C", "").split("-")
            low, high = float(low), float(high)
            if temp < low:
                return temp - low
            elif temp > high:
                return temp - high
            return 0.0
    except Exception:
        pass
    return 999.0
