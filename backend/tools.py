import random
import requests
from langchain_core.tools import tool
from urllib.parse import quote

from backend import settings


# Function to get car seat inspection stations from the NHTSA API
@tool
def get_seat_stations(state: str) -> str:
    """Get child car-seat inspection stations in a two-letter US state."""
    state = state.strip().upper()
    if len(state) != 2 or not state.isalpha():
        raise ValueError("State must be a two-letter abbreviation")
    request = requests.get(
        f"https://api.nhtsa.gov/CSSIStation/state/{quote(state, safe='')}",
        timeout=settings.nhtsa_timeout,
    )
    request.raise_for_status()
    results = request.json().get("Results", [])
    if not results:
        return f"No car seat inspection stations found in {state}."

    random.shuffle(results)
    summary = f"Car Seat Inspection Stations in {state}:\n"
    for station in results[:10]:
        summary += (
            f"\n- {station.get('Organization', 'N/A')} "
            f"{station.get('AddressLine1', 'N/A')} "
            f"{station.get('City', 'N/A')}, "
            f"{station.get('State', 'N/A')} {station.get('Zip', 'N/A')}"
        )
    return summary


# Function to get vehicle recalls from the NHTSA API
@tool
def get_recalls(make: str, model: str, year: str) -> str:
    """Get recalls for a vehicle given its make, model, and year."""
    if len(str(year)) != 4 or not str(year).isdigit():
        raise ValueError("Year must contain four digits")
    request = requests.get(
        "https://api.nhtsa.gov/recalls/recallsByVehicle",
        params={"make": make, "model": model, "modelYear": year},
        timeout=settings.nhtsa_timeout,
    )
    request.raise_for_status()
    results = request.json().get("results", [])
    if not results:
        return f"No recalls found for {year} {make} {model}."

    summary = f"Recalls for {year} {make} {model}:\n"
    for i, recall in enumerate(results, 1):
        summary += (
            f"\n{i}. {recall.get('Component', 'N/A')} : "
            f"{recall.get('Summary', 'No summary available')}"
        )
    return summary


# Function to get vehicle crash ratings from the NHTSA API
@tool
def get_crash_ratings(make: str, model: str, year: str) -> str:
    """Get NHTSA crash ratings for a vehicle given its make, model, and year."""
    if len(str(year)) != 4 or not str(year).isdigit():
        raise ValueError("Year must contain four digits")
    request = requests.get(
        "https://api.nhtsa.gov/SafetyRatings/"
        f"modelyear/{quote(str(year), safe='')}/"
        f"make/{quote(make, safe='')}/model/{quote(model, safe='')}",
        timeout=settings.nhtsa_timeout,
    )
    request.raise_for_status()
    vehicles = request.json().get("Results", [])
    if not vehicles:
        return f"No crash ratings found for {year} {make} {model}."

    summary = f"Crash Ratings for {year} {make} {model}:\n"
    for vehicle in vehicles:
        vehicle_id = vehicle.get("VehicleId")
        rating_request = requests.get(
            f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{vehicle_id}",
            timeout=settings.nhtsa_timeout,
        )
        rating_request.raise_for_status()
        results = rating_request.json().get("Results", [])
        if not results:
            continue
        rating = results[0]
        summary += f"\nVehicle ID: {vehicle_id}"
        summary += f"\n- Overall Rating: {rating.get('OverallRating', 'N/A')} stars"
        summary += f"\n- Front Crash: {rating.get('OverallFrontCrashRating', 'N/A')} stars"
        summary += f"\n- Side Crash: {rating.get('OverallSideCrashRating', 'N/A')} stars"
        summary += f"\n- Rollover: {rating.get('RolloverRating', 'N/A')} stars\n"
    return summary


# Function to decode a VIN before it is used as vehicle context
def decode_vin(vin: str) -> dict:
    vin = vin.strip().upper()
    if len(vin) != 17:
        raise ValueError("VIN must contain 17 characters")
    request = requests.get(
        "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/"
        f"{quote(vin, safe='')}",
        params={"format": "json"},
        timeout=settings.nhtsa_timeout,
    )
    request.raise_for_status()
    results = request.json().get("Results", [])
    if not results:
        return {}

    vehicle = results[0]
    return {
        "year": vehicle.get("ModelYear") or None,
        "make": vehicle.get("Make") or None,
        "model": vehicle.get("Model") or None,
        "vin": vin,
    }


tools = [get_recalls, get_crash_ratings, get_seat_stations]
tools_by_name = {selected_tool.name: selected_tool for selected_tool in tools}
