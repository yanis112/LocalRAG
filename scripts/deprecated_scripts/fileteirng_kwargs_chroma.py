
def get_filtering_kwargs_chromadb(
    source_filter: Union[list, str],
    source_filter_type: str,
    field_filter: Union[list, str],
    field_filter_type: str,
    length_threshold=None,
):
    """
    Get the kwargs that will be used to filter the search based on both the source and the field and chunk length.

    Args:
        source_filter (list or str): The source filter(s) to apply.
        source_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal, '$in' for inclusion, etc.).
        field_filter (list or str): The field filter(s) to apply.
        field_filter_type (str): The type of filter to apply ('$ne' for not equal, '$eq' for equal, '$in' for inclusion, etc.).
        length_threshold (float, optional): The minimum length of the chunks to return.

    Returns:
        dict: The updated search_kwargs dictionary with the source and field filters applied.
    """
    search_kwargs = {"where": {}}

    # Helper to create filter conditions
    def create_filter_condition(field, value, filter_type):
        return {field: {filter_type: value}}

    # Build filters based on input type for source
    if isinstance(source_filter, list):
        if source_filter_type == "$in":
            search_kwargs["where"]["metadata.source"] = {
                source_filter_type: source_filter
            }
        else:
            for source in source_filter:
                condition = create_filter_condition(
                    "metadata.source", source, source_filter_type
                )
                search_kwargs["where"].update(condition)
    else:
        condition = create_filter_condition(
            "metadata.source", source_filter, source_filter_type
        )
        search_kwargs["where"].update(condition)

    # Build filters based on input type for field
    if isinstance(field_filter, list):
        if field_filter_type == "$in":
            search_kwargs["where"]["metadata.field"] = {
                field_filter_type: field_filter
            }
        else:
            for field in field_filter:
                condition = create_filter_condition(
                    "metadata.field", field, field_filter_type
                )
                search_kwargs["where"].update(condition)
    else:
        condition = create_filter_condition(
            "metadata.field", field_filter, field_filter_type
        )
        search_kwargs["where"].update(condition)

    # Add length filter if length_threshold is specified
    if length_threshold is not None:
        search_kwargs["where"]["metadata.chunk_length"] = {
            "$gte": float(length_threshold)
        }

    print(
        "Searching for the following sources:",
        source_filter,
        "with filter type:",
        source_filter_type,
    )
    print(
        "Searching for the following fields:",
        field_filter,
        "with filter type:",
        field_filter_type,
    )
    return search_kwargs