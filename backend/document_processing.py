import hashlib
import re


known_makes = sorted(
    [
        "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BMW", "BUICK", "CADILLAC",
        "CHEVROLET", "CHRYSLER", "DODGE", "FIAT", "FORD", "GENESIS",
        "GMC", "HONDA", "HYUNDAI", "INFINITI", "JAGUAR", "JEEP", "KIA",
        "LAND ROVER", "LEXUS", "LINCOLN", "LUCID", "MAZDA", "MERCEDES-BENZ",
        "MERCEDES BENZ", "MINI", "MITSUBISHI", "NISSAN", "OLDSMOBILE",
        "PLYMOUTH", "POLESTAR", "PONTIAC", "PORSCHE", "RAM", "RIVIAN",
        "SAAB", "SATURN", "SCION", "SMART", "SUBARU", "SUZUKI", "TESLA",
        "TOYOTA", "VOLKSWAGEN", "VOLVO",
    ],
    key=len,
    reverse=True,
)

component_keywords = {
    "air bags": ("air bag", "airbag"),
    "brakes": ("brake", "abs", "stopping distance"),
    "electrical": ("electrical", "battery", "alternator", "wiring"),
    "engine": ("engine", "motor", "oil", "stall", "piston"),
    "fuel system": ("fuel", "gasoline", "diesel", "fuel pump"),
    "seat belts": ("seat belt", "seatbelt"),
    "steering": ("steering", "power steering"),
    "suspension": ("suspension", "strut", "shock absorber"),
    "tires and wheels": ("tire", "wheel", "rim", "tpms"),
    "transmission": ("transmission", "gear", "shifting", "shifted"),
    "visibility": ("windshield", "visibility", "wiper", "defroster"),
}


def normalize_document(document: str) -> str:
    return " ".join(document.split()).strip()


def get_stable_document_id(document: str) -> str:
    normalized = normalize_document(document).lower().encode("utf-8")
    return f"nhtsa_{hashlib.sha256(normalized).hexdigest()[:32]}"


def extract_vehicle_metadata(document: str) -> dict:
    normalized = normalize_document(document).upper()
    for match in re.finditer(r"\b((?:19|20)\d{2})\s+", normalized):
        remaining = normalized[match.end():]
        for make in known_makes:
            if not remaining.startswith(f"{make} "):
                continue
            model_text = remaining[len(make) + 1:]
            model_text = re.split(r"[.,;:]", model_text, maxsplit=1)[0]
            model_text = re.split(
                r"\b(?:THAT|WHICH|WITH|WHEN|WHILE|WHERE|THE CONTACT|WAS|IS)\b",
                model_text,
                maxsplit=1,
            )[0]
            model = " ".join(model_text.split()[:4]).strip(" -")
            if model:
                return {
                    "year": match.group(1),
                    "make": make.replace("MERCEDES BENZ", "MERCEDES-BENZ"),
                    "model": model,
                }
    return {}


def extract_component(document: str) -> str | None:
    text = document.lower()
    scores = {
        component: sum(text.count(keyword) for keyword in keywords)
        for component, keywords in component_keywords.items()
    }
    component, score = max(scores.items(), key=lambda item: item[1])
    return component if score else None


def create_document_metadata(document: str, source_id: str) -> dict:
    metadata = {
        "source": "nhtsa_complaint",
        "source_id": str(source_id),
        "schema_version": 2,
    }
    metadata.update(extract_vehicle_metadata(document))
    component = extract_component(document)
    if component:
        metadata["component"] = component
    return metadata
