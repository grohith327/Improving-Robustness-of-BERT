import os
import sys

dataset_name = sys.argv[1]
if dataset_name == "mr":
    max_seq_length = 128
    batch_size = 32
    sim_score = 0.68
else:
    max_seq_length = 256
    batch_size = 16
    sim_score = 0.89

command = (
    "python attack_classification.py --dataset_path data/{} "
    "--target_model bert "
    "--target_model_path trained_model/yelp "
    "--max_seq_length {} --batch_size {} --sim_score_threshold {} "
    "--counter_fitting_embeddings_path embeddings/counter_fitted_vectors.txt "
    "--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy "
    "--USE_cache_path ../tf_cache"
).format(dataset_name, max_seq_length, batch_size, sim_score)

os.system(command)
