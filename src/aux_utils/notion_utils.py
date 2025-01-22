import os
from notion_client import Client
import re
from dotenv import load_dotenv

load_dotenv()
class NotionScrapper:
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_API_KEY"))
        self.data_folder = "data/notion"
        os.makedirs(self.data_folder, exist_ok=True)

    def sanitize_filename(self, filename):
        """
        Sanitize a filename by replacing invalid characters with underscores.
        Args:
            filename (str): The filename to sanitize.
        Returns:
            str: The sanitized filename.
        """
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    def get_text_content(self, rich_text):
        """
        Extracts plain text content from a Notion rich text object.
        Args:
            rich_text (list): A list of rich text objects.
        Returns:
            str: The concatenated text content.
        """
        if not rich_text:
            return ""
        return "".join(text.get("text", {}).get("content", "") for text in rich_text)

    def handle_block(self, block):
        """
        Converts a Notion block to markdown format.
        Args:
            block (dict): A Notion block object.
        Returns:
            str: The block content in markdown format.
        """
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
        """
        Retrieves all child blocks of a given Notion block.
        Args:
            block_id (str): The ID of the parent Notion block.
        Returns:
            list: A list of child block objects.
        """
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
        """
        Downloads a Notion page and saves it as a markdown file.
        Args:
            page_id (str): The ID of the Notion page to download.
        Retrieves the content and metadata of the specified Notion page,
        converts it to markdown format, and saves it to a file in the
        specified data folder. The filename is derived from the page title.
        The function handles nested blocks by recursively retrieving and
        converting child blocks to markdown format.
        Raises:
            IOError: If there is an error writing the markdown file.
        """
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
        """
        Scrapes pages from Notion and downloads them.
        This method searches for pages in Notion using a filter and retrieves the search results.
        For each page found, it downloads the page using the `download_page` method.
        Prints the number of pages found and the status of each page download.
        Returns:
            None
        """
        search_results = self.notion.search(filter={"property": "object", "value": "page"}).get("results", [])
        print(f"Found {len(search_results)} pages.")
        
        for result in search_results:
            if result["object"] == "page":
                page_id = result["id"]
                print(f"Downloading page: {page_id}")
                self.download_page(page_id)
                print(f"Page downloaded: {page_id}")

    
    def create_page_from_markdown(self, markdown_content, page_title, parent_page_id = None):
        """
        Creates a new Notion page with content from markdown text.
        Args:
            markdown_content (str): The markdown text content for the page.
            page_title (str): The title of the new Notion page.
            parent_page_id (str, optional): The ID of the parent page, if the new page should be nested. Defaults to None.
        Returns:
            str: The ID of the newly created Notion page.
        
        Parses markdown into a list of Notion blocks, creates a new Notion page
        with the given title, and adds the parsed blocks as content.
        The function supports headings, paragraphs, lists, to-do lists, toggle,
        code, quotes, and dividers.
        Raises:
            Exception: If there is a error while creating the notion page
        """
        blocks = []
        lines = markdown_content.splitlines()
        i = 0
        while i < len(lines):
             line = lines[i].strip()
             if line.startswith('# '):
                blocks.append({
                   "type": "heading_1",
                   "heading_1": {
                       "rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]
                       }
                   })
             elif line.startswith('## '):
                blocks.append({
                   "type": "heading_2",
                   "heading_2": {
                       "rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]
                       }
                   })
             elif line.startswith('### '):
                  blocks.append({
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}]
                    }
                  })
             elif line.startswith('* '):
                  blocks.append({
                      "type": "bulleted_list_item",
                      "bulleted_list_item": {
                          "rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]
                      }
                  })
             elif line.startswith('1. '):
                 blocks.append({
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]
                    }
                  })
             elif line.startswith('- ['):
                 is_checked = line[3] == 'x'
                 text_content = line[6:].strip()
                 blocks.append({
                     "type": "to_do",
                     "to_do":{
                         "rich_text": [{"type": "text", "text": {"content": text_content}}],
                         "checked": is_checked
                    }
                })
             elif line.startswith('<details>'):
                 i += 1
                 summary_line = lines[i].strip()
                 if not summary_line.startswith('<summary>'):
                     raise ValueError(f"invalid toggle format: {summary_line}")
                 summary_text = summary_line[9:-10].strip()

                 i+= 1
                 content_lines = []
                 while i < len(lines) and not lines[i].strip().startswith('</details>'):
                     content_lines.append(lines[i])
                     i+=1
                 content_text = "\n".join(content_lines).strip()

                 blocks.append({
                    "type":"toggle",
                    "toggle":{
                       "rich_text": [{"type":"text", "text":{"content": summary_text}}],
                         }
                    })
                 blocks.extend(self.markdown_to_blocks(content_text))
                 
             elif line.startswith('```'):
                 code_language = line[3:].strip()
                 i+=1
                 code_lines = []
                 while i < len(lines) and not lines[i].strip().startswith('```'):
                     code_lines.append(lines[i])
                     i+=1
                 code_text = "\n".join(code_lines)
                 blocks.append({
                     "type": "code",
                     "code": {
                        "language": code_language,
                        "rich_text": [{"type": "text", "text": {"content": code_text}}]
                     }
                    })
             elif line.startswith('> '):
                 blocks.append({
                     "type":"quote",
                     "quote": {
                         "rich_text":[{"type":"text", "text": {"content": line[2:].strip()}}]
                     }
                 })
             elif line == '---':
                blocks.append({"type": "divider"})
             elif line:
                blocks.append({
                   "type": "paragraph",
                   "paragraph": {
                       "rich_text": [{"type": "text", "text": {"content": line}}]
                       }
                    })
             i+=1

        page_properties = {
           "title": {
                "title": [{"type": "text", "text": {"content": page_title}}]
            }
        }

        if parent_page_id:
           if not isinstance(parent_page_id, str) or not re.match(r"^[0-9a-f]{32}$", parent_page_id):
               raise ValueError("parent_page_id must be a valid 32 character long Notion page ID")
           parent = {"type": "page_id", "page_id": parent_page_id}
        else:
            # Use workspace as parent
            parent = {"type": "workspace"}
        try:
            new_page = self.notion.pages.create(parent=parent, properties=page_properties, children=blocks)
        except Exception as e:
            raise Exception(f"Error while creating a new page in the database: {e}")
        return new_page["id"]
    
    def markdown_to_blocks(self, markdown_content):
        """
         Converts markdown to a list of Notion block objects for create_page_from_markdown
         Args:
            markdown_content (str): markdown text
         Returns:
           list: list of notion block objects 
        """
        blocks = []
        lines = markdown_content.splitlines()
        i = 0
        while i < len(lines):
             line = lines[i].strip()
             if line.startswith('# '):
                blocks.append({
                   "type": "heading_1",
                   "heading_1": {
                       "rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]
                       }
                   })
             elif line.startswith('## '):
                blocks.append({
                   "type": "heading_2",
                   "heading_2": {
                       "rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]
                       }
                   })
             elif line.startswith('### '):
                  blocks.append({
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}]
                    }
                  })
             elif line.startswith('* '):
                  blocks.append({
                      "type": "bulleted_list_item",
                      "bulleted_list_item": {
                          "rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]
                      }
                  })
             elif line.startswith('1. '):
                 blocks.append({
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]
                    }
                  })
             elif line.startswith('- ['):
                 is_checked = line[3] == 'x'
                 text_content = line[6:].strip()
                 blocks.append({
                     "type": "to_do",
                     "to_do":{
                         "rich_text": [{"type": "text", "text": {"content": text_content}}],
                         "checked": is_checked
                    }
                })
             elif line.startswith('<details>'):
                 i += 1
                 summary_line = lines[i].strip()
                 if not summary_line.startswith('<summary>'):
                     raise ValueError(f"invalid toggle format: {summary_line}")
                 summary_text = summary_line[9:-10].strip()

                 i+= 1
                 content_lines = []
                 while i < len(lines) and not lines[i].strip().startswith('</details>'):
                     content_lines.append(lines[i])
                     i+=1
                 content_text = "\n".join(content_lines).strip()

                 blocks.append({
                    "type":"toggle",
                    "toggle":{
                       "rich_text": [{"type":"text", "text":{"content": summary_text}}],
                         }
                    })
                 blocks.extend(self.markdown_to_blocks(content_text))
                 
             elif line.startswith('```'):
                 code_language = line[3:].strip()
                 i+=1
                 code_lines = []
                 while i < len(lines) and not lines[i].strip().startswith('```'):
                     code_lines.append(lines[i])
                     i+=1
                 code_text = "\n".join(code_lines)
                 blocks.append({
                     "type": "code",
                     "code": {
                        "language": code_language,
                        "rich_text": [{"type": "text", "text": {"content": code_text}}]
                     }
                    })
             elif line.startswith('> '):
                 blocks.append({
                     "type":"quote",
                     "quote": {
                         "rich_text":[{"type":"text", "text": {"content": line[2:].strip()}}]
                     }
                 })
             elif line == '---':
                blocks.append({"type": "divider"})
             elif line:
                blocks.append({
                   "type": "paragraph",
                   "paragraph": {
                       "rich_text": [{"type": "text", "text": {"content": line}}]
                       }
                    })
             i+=1
        return blocks

if __name__ == "__main__":
    import asyncio
   
    scrapper = NotionScrapper()
        # Example of scraping pages
        #scrapper.scrape()
        # Example of creating a page from markdown
    markdown_text = """
    # My New Page Title

    This is a paragraph of text.

    ## A Subheading

    - Bullet point 1
    - Bullet point 2
        - Nested bullet
        
    1. Numbered list item 1
    2. Numbered list item 2
        
    - [ ] To do item 1
    - [x] To do item 2

    <details>
    <summary>Toggle Summary</summary>

    this is the content of a toggle

    </details>

    ```python
    print("Hello, world!")"""
 
    #try creating this new page
    new_page_id = scrapper.create_page_from_markdown(markdown_text, "My New Page Title")
    print("new page id : ", new_page_id)