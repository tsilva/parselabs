from utils import generate_lab_spec_names_embeddings_cache, generate_lab_name_mappings, load_json, save_json

generate_lab_spec_names_embeddings_cache()

labs = load_json("outputs/labs_results.final.json")
mappings = generate_lab_name_mappings(labs)
save_json("outputs/mappings.json", mappings)