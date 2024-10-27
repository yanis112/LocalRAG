
@log_execution_time
def clone_database_chunks(
    clone_persist, clone_embedding_model, persist_directory
):
    print("CLONING DATABASE!")
    chunks, metadata = get_k_random_chunks(
        k=1,
        config=None,
        get_all=True,
        clone_persist=clone_persist,
        clone_embedding_model=clone_embedding_model,
    )
    total_chunks = [
        Document(
            page_content=chunk,
            metadata={
                "source": metadata[i]["source"],
                "chunk_length": metadata[i]["chunk_length"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    print("CLONING DONE!")
    json_files = [f for f in os.listdir(clone_persist) if f.endswith(".json")]
    if json_files:
        json_file_path = os.path.join(clone_persist, json_files[0])
        base_name = os.path.basename(persist_directory.rstrip("/"))
        new_file_name = str(base_name) + "_evaluation_dataset.json"
        new_file_path = os.path.join(persist_directory, new_file_name)
        shutil.copy(json_file_path, new_file_path)
        print(
            f"Evaluation dataset json file copied to the persist directory as {new_file_name}!"
        )
    return total_chunks