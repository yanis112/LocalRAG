import os
from notion_client import Client
import re

class NotionScrapper:
    def __init__(self, token):
        self.notion = Client(auth=token)
        self.data_folder = "data/notion"
        os.makedirs(self.data_folder, exist_ok=True)

    def sanitize_filename(self, filename):
        # Remove or replace invalid characters for Windows file names
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    def download_page(self, page_id):
        # Retrieve page content
        page_content = self.notion.blocks.children.list(block_id=page_id)
        
        # Retrieve page metadata
        page_metadata = self.notion.pages.retrieve(page_id=page_id)
        
        # Combine content and metadata
        full_page = {
            "metadata": page_metadata,
            "content": page_content
        }
        
        # Save the page in markdown format
        page_title = page_metadata["properties"]["title"]["title"][0]["text"]["content"]
        sanitized_title = self.sanitize_filename(page_title)
        file_path = os.path.join(self.data_folder, f"{sanitized_title}.md")
        
        print(f"Saving page '{page_title}' to '{file_path}'")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.convert_to_markdown(full_page))

    def convert_to_markdown(self, full_page):
        # Convert the full page content to markdown format
        # This is a simplified example, you may need to handle different block types
        markdown_content = f"# {full_page['metadata']['properties']['title']['title'][0]['text']['content']}\n\n"
        for block in full_page["content"]["results"]:
            print(f"Processing block: {block}")
            if block["type"] == "paragraph":
                if "rich_text" in block["paragraph"] and block["paragraph"]["rich_text"]:
                    for text in block["paragraph"]["rich_text"]:
                        if "text" in text and "content" in text["text"]:
                            markdown_content += text["text"]["content"] + "\n\n"
                        else:
                            print(f"Skipping text due to missing 'content': {text}")
                else:
                    print(f"Skipping block due to missing 'rich_text': {block}")
            else:
                print(f"Skipping block of type '{block['type']}': {block}")
        return markdown_content

    def scrapo(self):
        # Search for all pages
        search_results = self.notion.search(filter={"property": "object", "value": "page"}).get("results")
        #Search things thare are not pages
        other_search_results = self.notion.search().get("results")
        print(f"Found {len(other_search_results)} other things.")
        
        print(f"Found {len(search_results)} pages.")
        
        for result in search_results:
            if result["object"] == "page":
                page_id = result["id"]
                print(f"Downloading page: {page_id}")
                self.download_page(page_id)
                print(f"Page downloaded: {page_id}")

if __name__ == "__main__":
    token = "secret_nOS3VbDLoHawArgBdmAQDRfueWLruEDloCVCKcUGalW"
    scrapper = NotionScrapper(token)
    scrapper.scrapo()