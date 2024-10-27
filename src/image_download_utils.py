from bing_image_downloader import downloader

query_string = 'neural network'
downloader.download(query_string, limit=5,  output_dir='dataset_image', 
adult_filter_off=True, force_replace=False, timeout=60)