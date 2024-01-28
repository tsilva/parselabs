import os
from tqdm import tqdm
from slugify import slugify
from dotenv import load_dotenv
from utils import load_json, save_json, create_completion

load_dotenv()

def generate_range_for_lab_unit(name: str, unit: str):
    USER_GENDER = os.environ["USER_GENDER"]
    USER_RACE = os.environ["USER_RACE"]
    USER_AGE = os.environ["USER_AGE"]
    USER_LOCATION = os.environ["USER_LOCATION"]
    USER_HEIGHT = os.environ["USER_HEIGHT"]
    USER_WEIGHT = os.environ["USER_WEIGHT"]

    system_prompt = f"""
Given a lab test corresponding measurement unit, what is the min and max healthy range for a person with the following characteristics:

Gender: {USER_GENDER}
Age: {USER_AGE}
Race: {USER_RACE}
Location: {USER_LOCATION}
Height: {USER_HEIGHT}
Weight: {USER_WEIGHT}

Output just the range as CSV string with range min value followed by range max value. If there is no min output NULL in its place, if there is no max output NULL in its place. Don't output anything else.

Eg:
USER: Vitamina E [UNIT = mg/L]
ASSISTANT: 5.5;18.5
USER: Vitamina E  [UNIT = µg/mL]
ASSISTANT: 2.8;18.4
USER: Vitamina E  [UNIT = µmol/L]
ASSISTANT: 12;42
""".strip()
    user_prompt = f"{name} [UNIT = {unit}]"
    message = create_completion(user_prompt, model="gpt-4-1106-preview", system_prompt=system_prompt)
    range_s = message.content.strip()
    range_values = range_s.replace("<", "").replace(">", "").replace(",", ".").split(";")
    range_values = [value.strip().lower() for value in range_values]
    if "negative" in range_values or "positive" in range_values: return 0, 1
    range_min, range_max = [float(value) if value != "NULL" else None for value in range_values]
    return range_min, range_max

def generate_units_for_lab_name(name):
    system_prompt = """
Given a lab test, what are the most 3 most common units used to measure that lab test (by default consider that it's a blood lab, unless specified otherwise). Output as CSV string, ordered from most to least common, with no other content.

eg: 
USER: Leucograma - Eosinófilos
ASSISTANT: %;células/µL;células/mm³
""".strip()
    message = create_completion(name, system_prompt=system_prompt)
    units_s = message.content.strip()
    units = units_s.split(";")
    units = [unit.strip() for unit in units if unit.strip()]
    return units

def generate_units_for_lab_specs(lab_specs):
    for spec in tqdm(lab_specs, total=len(lab_specs)):
        name = spec["name"]
        units = spec.get("units")
        if units: continue
        units = generate_units_for_lab_name(name)
        spec["units"] = dict([(unit, {}) for unit in units])
    return lab_specs

def generate_units_ranges_for_lab_specs(lab_specs):
    for spec in tqdm(lab_specs, total=len(lab_specs) * 3):
        lab_name = spec["name"]
        units = spec.get("units", {})
        for unit_name, values in units.items():
            if values: 
                continue
            try: 
                range_min, range_max = generate_range_for_lab_unit(lab_name, unit_name)
            except Exception as e: 
                print(e)
                continue
            spec["units"][unit_name] = {
                "min": range_min,
                "max": range_max
            }
            print(unit_name)
            print(spec["units"][unit_name])
    return lab_specs

def sort_labs_specs(lab_specs):
    return sorted(lab_specs, key=lambda spec: spec["name"])

def assert_unique_lab_specs(lab_specs):
    duplicates = []
    names_map = {}
    for lab_spec in lab_specs:
        name = lab_spec["name"]
        slug = slugify(name).replace("-", "")
        _names = names_map.get(slug, [])
        _names.append(name)
        names_map[slug] = _names
    for slug, names in names_map.items():
        if len(names) <= 1: continue
        duplicates.extend(names)

    if duplicates:
        raise Exception("Duplicate lab specs found: " + ", ".join(duplicates))

def extract_unique_units(lab_specs):
    VALID_UNITS = [
        "ng/dL", "ng/mL", "µg/L", "pg/mL", "pmol/L", "UI/mL", "U/mL", "AU/mL", "mg/dL",
        "UI/L", "UI/dL", "U/L", "AI", "RU/mL", "IU/mL", "IU/L", "g/dL", "g/L", "mg/L",
        "µmol/L", "nmol/L", "mmol/L", "µg/dL", "µg/mL", "fL", "pg", "µm³", "%",
        "células/µL", "células/mm³", "s", "µmol/dL", "Index", "µg/g", "mg/kg", "mEq/L",
        "mUI/mL", "COI", "S/CO", "ratio", "UFC/mL", "UFC/100mL", "UFC/µL", "kU/L",
        "AIU/mL", "mm/h", "mg/100mL", "nmol/dL", "segundos", "ug/g", "N/A", "nmol/mL",
        "µmol/g", "mIU/mL", "ng/L", "fl", "INR", "µU/mL", "µIU/mL", "mmol/g Hb",
        "10^3/µL", "mIU/L", "fl%", "10^9/L", "mU/L", "mg/24h", "mm³", "mL", "L", "µL",
        "g/mL", "kg/m³", "g/cm³", "g/24h", "células/campo", "células/mL", 
        "µkat/L", "kat/L", "10^6/µL", "10^12/L", "10^6/mm³", "GPLU/ML", "MPLU/ml"
    ]

    unique_units = {}
    invalid_units = {}
    for lab_spec in lab_specs:
        _units = lab_spec.get("units", {})
        for unit_name in _units.keys(): 
            unique_units[unit_name] = True
            if not unit_name in VALID_UNITS: invalid_units[unit_name] = True
    unique_units = list(unique_units.keys())
    invalid_units = list(invalid_units.keys())
    return {"unique" : unique_units, "invalid": invalid_units}

labs_specs = load_json("labs_specs.json")
assert_unique_lab_specs(labs_specs)
sort_labs_specs(labs_specs)
generate_units_for_lab_specs(labs_specs)
generate_units_ranges_for_lab_specs(labs_specs)
units = extract_unique_units(labs_specs)
save_json("labs_specs_2.units.json", units)
save_json("labs_specs_2.json", labs_specs)
