from utils import load_json, augment_lab_result, save_json

results = load_json("outputs/labs.json")
for result in results: augment_lab_result(result)
save_json("outputs/labs_fixed.json", results)
