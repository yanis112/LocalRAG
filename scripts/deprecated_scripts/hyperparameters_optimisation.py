import yaml
from bayes_opt import BayesianOptimization

# custom imports
from evaluate_pipeline import launch_eval


# Step 1: Define a function to generate config files
def generate_config(params, base_config_path):
    # Load existing config file
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Take a base config as input and create a copy of it with some of the parameters changed
    for key in params:
        if key in base_config:
            base_config[key] = params[key]
    with open("config/optimiz_config.yaml", "w") as f:
        yaml.dump(base_config, f)


# Define a simple polynomial function to replace launch_eval
""" def launch_eval(config):
    a=-config['nb_chunks']**3 - config['nb_rerank']**2 + config['length_threshold']**2
    return {'accuracy_top1': a}
 """


# Step 2: Define the objective function
def objective(nb_chunks, nb_rerank, length_threshold, use_multi_query):
    params = {
        "nb_chunks": int(nb_chunks),
        "nb_rerank": int(nb_rerank),
        "length_threshold": int(length_threshold),
        "use_multi_query": True if use_multi_query > 0.5 else False,
    }
    # Generate a config file
    generate_config(params, base_config_path="config/config.yaml")
    # Load the config file
    with open("config/optimiz_config.yaml") as f:
        config = yaml.safe_load(f)
    # Call the evaluation function
    metrics = launch_eval(config, evaluate_generation=False)
    # Return the metric to optimize
    return metrics["accuracy_top1"]


""" # Step 2: Define the objective function
def objective(nb_chunks, nb_rerank, top_k, chunk_size, chunk_overlap, length_threshold, auto_merging_threshold, num_exemples, num_queries):
    # Here we define a simple polynomial to optimize
    return -(nb_chunks**2 + nb_rerank**2 + top_k**2 + chunk_size**2 + chunk_overlap**2 + length_threshold**2 + auto_merging_threshold**2 + num_exemples**2 + num_queries**2)
 """


if __name__ == "__main__":
    e = generate_config(
        params={"nb_chunks": 5}, base_config_path="config/config.yaml"
    )

    # exit()

    # Step 3: Use Bayesian optimization
    pbounds = {
        "nb_chunks": (5, 30),
        "nb_rerank": (5, 20),
        "length_threshold": (1, 30),
        "use_multi_query": (0, 1),  # categorial true fale !
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=2, n_iter=5)

    print("########################################")
    print("OPTIMIZED PARAMETERS:", optimizer.max)
