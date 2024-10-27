
@log_execution_time
def apply_auto_hybrid_search(query):
    """
    Applies auto hybrid search to the given query.

    Args:
        query (str): The query to be processed.

    Returns:
        tuple: A tuple containing the hybrid search flag and the word filter.
            - hybrid_search (bool): True if hybrid search is enabled, False otherwise.
            - word_filter (str): The word filter extracted for lexical queries, None for semantic queries.
    """
    print("AUTO HYBRID SEARCH TRIGGERED !")
    query_router = QueryRouter()
    query_router.load()
    label = query_router.predict_label(query)
    hybrid_search = False
    word_filter = None
    if label == "semantic":
        print("QUERY CLASSIFIED AS SEMANTIC !")
    else:
        print("QUERY CLASSIFIED AS LEXICAL, ENABLING HYBRID SEARCH !")
        hybrid_search = True
        word_filter = keywords_extraction(query)
        word_filter = word_filter[0][0]
        print("AUTOMATIC EXTRACTED WORD FILTER:", str(word_filter))
    return hybrid_search, word_filter