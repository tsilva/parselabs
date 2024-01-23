from utils import generate_units_for_lab_specs, generate_units_ranges_for_lab_specs, load_json, save_json

#generate_lab_spec_names_embeddings_cache()

labs_specs = load_json("labs_specs.json")
generate_units_for_lab_specs(labs_specs)
generate_units_ranges_for_lab_specs(labs_specs)
save_json("labs_specs_2.json", labs_specs)