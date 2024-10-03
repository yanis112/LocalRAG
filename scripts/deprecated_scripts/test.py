from src.generation_utils import LLM_answer_v3


def truncate_path_to_data(path):
    """
    Truncates the given path to start from '/data' if '/data' is present in the path.

    Args:
        path (str): The original file path.

    Returns:
        str: The truncated or original path.
    """
    # Check if '/data' is in the path
    if 'data/' in path:
        # Find the index of '/data' and truncate everything before it
        data_index = path.index('data/')
        return path[data_index:]
    else:
        # Return the original path if '/data' is not found
        return path


if __name__ == "__main__":
    # Example usage
    original_path1 = "/home/user/llms/data/chatbot_history/Qui_sont_les_employés_de_Euranova_qui_travaillent_sur_Digazu_.json"
    original_path2 = "data/pages/1gqes6r3j28el6kb88?locale=no_language&q=%23onboarding.html"

    print(truncate_path_to_data(original_path1))
    print(truncate_path_to_data(original_path2))