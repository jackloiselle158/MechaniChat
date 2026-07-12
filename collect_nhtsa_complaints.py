import requests
import csv
import time

# 20 popular makes and models (common U.S. vehicles)
vehicles = [
    ("TOYOTA", "CAMRY"),
    ("TOYOTA", "COROLLA"),
    ("HONDA", "CIVIC"),
    ("HONDA", "ACCORD"),
    ("FORD", "F-150"),
    ("FORD", "ESCAPE"),
    ("CHEVROLET", "SILVERADO"),
    ("CHEVROLET", "MALIBU"),
    ("NISSAN", "ALTIMA"),
    ("NISSAN", "ROGUE"),
    ("JEEP", "GRAND CHEROKEE"),
    ("JEEP", "WRANGLER"),
    ("HYUNDAI", "ELANTRA"),
    ("HYUNDAI", "SONATA"),
    ("KIA", "OPTIMA"),
    ("KIA", "SOUL"),
    ("VOLKSWAGEN", "JETTA"),
    ("VOLKSWAGEN", "PASSAT"),
    ("SUBARU", "OUTBACK"),
    ("SUBARU", "FORESTER")
]

years = range(2010, 2023)

complaints = []

for make, model in vehicles:
    for year in years:
        url = f"https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={year}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for item in results:
                    # complaint text may be under "summary" or "text"
                    desc = item.get("summary") or item.get("text") or ""
                    if desc.strip():
                        complaints.append([desc.strip()])
                print(f"Finished {make} {model} {year}.")
            else:
                print(f"⚠️ Skipped {make} {model} {year}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"❌ Error with {make} {model} {year}: {e}")

        # Be polite to the API (avoid hammering their servers)
        time.sleep(0.5)

print(f"✅ Collected {len(complaints)} complaints.")

# Save to CSV
with open("nhtsa_complaints.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["complaint"])
    writer.writerows(complaints)

# Save to TXT
with open("nhtsa_complaints.txt", "w", encoding="utf-8") as f:
    for c in complaints:
        f.write(c[0] + "\n")

print("💾 Saved to nhtsa_complaints.csv and nhtsa_complaints.txt")
