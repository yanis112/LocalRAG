import os
from notion_client import Client
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
class NotionScrapper:
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_API_KEY"))
        self.data_folder = "data/notion"
        os.makedirs(self.data_folder, exist_ok=True)

    def sanitize_filename(self, filename):
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    def get_text_content(self, rich_text):
        if not rich_text:
            return ""
        return "".join(text.get("text", {}).get("content", "") for text in rich_text)

    def handle_block(self, block):
        block_type = block["type"]
        content = ""

        if block_type == "paragraph":
            text = self.get_text_content(block["paragraph"].get("rich_text", []))
            content = f"{text}\n\n" if text else "\n"
            
        elif block_type == "heading_1":
            text = self.get_text_content(block["heading_1"].get("rich_text", []))
            content = f"# {text}\n\n"
            
        elif block_type == "heading_2":
            text = self.get_text_content(block["heading_2"].get("rich_text", []))
            content = f"## {text}\n\n"
            
        elif block_type == "heading_3":
            text = self.get_text_content(block["heading_3"].get("rich_text", []))
            content = f"### {text}\n\n"
            
        elif block_type == "bulleted_list_item":
            text = self.get_text_content(block["bulleted_list_item"].get("rich_text", []))
            content = f"* {text}\n"
            
        elif block_type == "numbered_list_item":
            text = self.get_text_content(block["numbered_list_item"].get("rich_text", []))
            content = f"1. {text}\n"
            
        elif block_type == "to_do":
            text = self.get_text_content(block["to_do"].get("rich_text", []))
            checked = "x" if block["to_do"].get("checked", False) else " "
            content = f"- [{checked}] {text}\n"
            
        elif block_type == "toggle":
            text = self.get_text_content(block["toggle"].get("rich_text", []))
            content = f"<details>\n<summary>{text}</summary>\n\n"
            
        elif block_type == "code":
            text = self.get_text_content(block["code"].get("rich_text", []))
            language = block["code"].get("language", "")
            content = f"```{language}\n{text}\n```\n\n"
            
        elif block_type == "quote":
            text = self.get_text_content(block["quote"].get("rich_text", []))
            content = f"> {text}\n\n"
            
        elif block_type == "divider":
            content = "---\n\n"
            
        return content

    async def get_block_children(self, block_id):
        children = []
        cursor = None
        
        while True:
            response = await self.notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
            children.extend(response["results"])
            
            if not response["has_more"]:
                break
                
            cursor = response["next_cursor"]
            
        return children

    def download_page(self, page_id):
        page_content = self.notion.blocks.children.list(block_id=page_id)
        page_metadata = self.notion.pages.retrieve(page_id=page_id)
        
        title = page_metadata["properties"].get("title", {}).get("title", [{}])[0].get("text", {}).get("content", "Untitled")
        sanitized_title = self.sanitize_filename(title)
        file_path = os.path.join(self.data_folder, f"{sanitized_title}.md")
        
        print(f"Saving page '{title}' to '{file_path}'")
        
        markdown_content = f"# {title}\n\n"
        for block in page_content["results"]:
            markdown_content += self.handle_block(block)
            
            # Handle nested blocks
            if "has_children" in block and block["has_children"]:
                children = self.notion.blocks.children.list(block_id=block["id"])
                for child in children["results"]:
                    markdown_content += "    " + self.handle_block(child).replace("\n", "\n    ")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def scrape(self):
        search_results = self.notion.search(filter={"property": "object", "value": "page"}).get("results", [])
        print(f"Found {len(search_results)} pages.")
        
        for result in search_results:
            if result["object"] == "page":
                page_id = result["id"]
                print(f"Downloading page: {page_id}")
                self.download_page(page_id)
                print(f"Page downloaded: {page_id}")

if __name__ == "__main__":
    scrapper = NotionScrapper()
    scrapper.scrape()