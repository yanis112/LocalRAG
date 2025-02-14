import os
from notion_client import Client
import re
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class NotionAgent:
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

    def create_page_from_markdown(self, markdown_content: str, page_title: str) -> str:
        """
        Creates a new page in Notion from the given markdown content.

        Args:
            markdown_content (str): The markdown content to convert into Notion blocks.
            page_title (str): The title of the new page.

        Returns:
            str: The ID of the newly created Notion page.
        
        Raises:
            Exception: If there is an error creating the page in Notion.
        """
        blocks = self.markdown_to_blocks(markdown_content)

        # To create a page at the top level, use the "workspace" parent type.
        #parent = {"type": "workspace", "workspace": True}
        parent = {"type": "database_id", "database_id": os.getenv("MOTHER_NOTION_PAGE_ID")}
        
        

        try:
            new_page = self.notion.pages.create(
                parent=parent,
                properties={
                    # Use the provided page title in the properties.
                    "title": {
                        "title": [{"type": "text", "text": {"content": page_title}}]
                    }
                },
                children=blocks
            )
            return new_page["id"]
        except Exception as e:
            raise Exception(f"Erreur de création de page: {str(e)}")

    def markdown_to_blocks(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Converts a markdown string to a list of Notion blocks.
        
        Args:
            markdown_content (str): The markdown content to convert.
        
        Returns:
            List[Dict[str, Any]]: A list of Notion blocks.
        """
        blocks = []
        lines = markdown_content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            raw_line = lines[i]

            # Gestion des titres
            if line.startswith('### '):
                blocks.append(self._create_heading_block(line, 3))
                i += 1
            elif line.startswith('## '):
                blocks.append(self._create_heading_block(line, 2))
                i += 1
            elif line.startswith('# '):
                blocks.append(self._create_heading_block(line, 1))
                i += 1

            # Listes à puces
            elif line.startswith('* ') or line.startswith('- '):
                list_blocks, i = self._handle_list(lines, i, "bulleted")
                blocks.extend(list_blocks)
            
            # Listes numérotées
            elif re.match(r'^\d+\. ', line):
                list_blocks, i = self._handle_list(lines, i, "numbered")
                blocks.extend(list_blocks)
            
            # Checklist
            elif line.startswith('- [ ] ') or line.startswith('- [x] '):
                blocks.append(self._create_todo_block(line))
                i += 1
            
            # Code blocks
            elif line.startswith('```'):
                blocks.append(self._create_code_block(lines, i))
                i = self._skip_code_block(lines, i)
            
            # Citation
            elif line.startswith('> '):
                blocks.append(self._create_quote_block(lines, i))
                i += 1
            
            # Divider
            elif line == '---':
                blocks.append({"object": "block", "type": "divider", "divider": {}})
                i += 1
            
            # Toggle (détails)
            elif line.startswith('<details>'):
                toggle_block, i = self._create_toggle_block(lines, i)
                blocks.append(toggle_block)
            
             # Tableaux
            if line.startswith('|'):
                table_block, i = self._handle_table(lines, i)
                if table_block:  # Check if _handle_table returned a valid block
                    blocks.append(table_block)
            
            # Paragraphe standard
            elif line:
                paragraph_block = self._create_paragraph_block(raw_line)
                blocks.append(paragraph_block)
                i += 1
            
            else:
                i += 1

        return blocks
    
    def _handle_table(self, lines: List[str], index: int) -> tuple:
        """
        Handles markdown tables and converts them to Notion table blocks.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.

        Returns:
            tuple: A tuple containing the Notion table block and the updated index.
        """
        table_lines = []
        while index < len(lines) and lines[index].strip().startswith('|'):
            table_lines.append(lines[index].strip())
            index += 1

        if len(table_lines) < 2:  # Not a valid table
            return None, index

        # Parse table data
        header_row = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
        num_cols = len(header_row)
        table_rows = []

        for row in table_lines[2:]:  # Skip header and separator lines
            row_data = [cell.strip() for cell in row.split('|') if cell.strip()]
            # Handle rows with incorrect number of columns
            if len(row_data) != num_cols:
                print(f"Warning: Skipping row with incorrect number of columns: {row_data}")
                continue
            # CORRECTION ICI : Chaque cellule doit être dans une liste
            table_rows.append([[{ "type": "text", "text": { "content": cell } }] for cell in row_data])

        # Create Notion table block
        table_block = {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": num_cols,
                "has_column_header": True,
                "has_row_header": False,
                "children": [
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {
                            # CORRECTION ICI : Chaque cellule doit être dans une liste
                            "cells": [[{ "type": "text", "text": { "content": cell } }] for cell in header_row]
                        }
                    }
                ]
            }
        }

        # Add table rows to the table block
        for row in table_rows:
            table_block["table"]["children"].append(
                {
                    "object": "block",
                    "type": "table_row",
                    "table_row": {
                        "cells": row
                    }
                }
            )

        return table_block, index

    def _create_heading_block(self, line: str, level: int) -> Dict:
        """
        Creates a Notion heading block.

        Args:
            line (str): The line containing the heading text.
            level (int): The heading level (1, 2, or 3).

        Returns:
            Dict: A Notion heading block.
        """
        return {
            "object": "block",
            "type": f"heading_{level}",
            f"heading_{level}": {
                "rich_text": [{"type": "text", "text": {"content": line[level+1:].strip()}}]
            }
        }

    def _handle_list(self, lines: List[str], index: int, list_type: str) -> tuple:
        """
        Handles bulleted and numbered lists.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.
            list_type (str): The type of list ("bulleted" or "numbered").

        Returns:
            tuple: A tuple containing the list of Notion blocks and the updated index.
        """
        items = []
        first_line = lines[index]
        base_indent = len(first_line) - len(first_line.lstrip())

        while index < len(lines):
            line = lines[index]
            indent = len(line) - len(line.lstrip())

            if indent < base_indent or not line.strip():
                break  # Fin de la liste

            content = line.strip().lstrip('*-1234567890. ').strip()

            if indent == base_indent:  # Même niveau
                items.append({
                    "type": f"{list_type}_list_item",
                    "content": content,
                    "children": []
                })
            else:  # Sous-niveau
                if not items:  # Pas de parent, on crée un au même niveau
                    items.append({
                        "type": f"{list_type}_list_item",
                        "content": "",
                        "children": []
                    })
                items[-1]["children"].append({
                    "type": f"{list_type}_list_item",
                    "content": content
                })

            index += 1

        return self._build_list_blocks(items, list_type), index

    def _build_list_blocks(self, items: List[Dict], list_type: str) -> List[Dict]:
        """
        Builds a nested list structure for Notion blocks, handling children.

        Args:
            items (List[Dict]): A list of list items, potentially nested.
            list_type (str): The type of list ("bulleted" or "numbered").

        Returns:
            List[Dict]: A list of Notion blocks representing the nested list.
        """
        blocks = []
        for item in items:
            block = {
                "object": "block",
                "type": item["type"],
                item["type"]: {
                    "rich_text": [{"type": "text", "text": {"content": item["content"]}}]
                }
            }
            if "children" in item and item["children"]:
                block[item["type"]]["children"] = self._build_list_blocks(item["children"], list_type)
            blocks.append(block)
        return blocks

    def _create_todo_block(self, line: str) -> Dict:
        """
        Creates a Notion to-do block.

        Args:
            line (str): The line containing the to-do item.

        Returns:
            Dict: A Notion to-do block.
        """
        checked = 'x' in line
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": line[5:].strip()}}],
                "checked": checked
            }
        }

    def _create_code_block(self, lines: List[str], index: int) -> Dict:
        """
        Creates a Notion code block.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.

        Returns:
            Dict: A Notion code block.
        """
        language = lines[index][3:].strip()
        code_lines = []
        index += 1
        
        while index < len(lines) and not lines[index].strip().startswith('```'):
            code_lines.append(lines[index])
            index += 1
        
        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": '\n'.join(code_lines)}}],
                "language": language if language else "plain text"
            }
        }

    def _skip_code_block(self, lines: List[str], index: int) -> int:
        """
        Skips the lines of a code block.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.

        Returns:
            int: The updated index after skipping the code block.
        """
        while index < len(lines) and not lines[index].strip().startswith('```'):
            index += 1
        return index + 1

    def _create_quote_block(self, lines: List[str], index: int) -> Dict:
        """
        Creates a Notion quote block.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.

        Returns:
            Dict: A Notion quote block.
        """
        quote_lines = []
        while index < len(lines) and lines[index].strip().startswith('> '):
            quote_lines.append(lines[index].lstrip('> ').strip())
            index += 1
        return {
            "object": "block",
            "type": "quote",
            "quote": {
                "rich_text": [{"type": "text", "text": {"content": ' '.join(quote_lines)}}]
            }
        }

    def _create_toggle_block(self, lines: List[str], index: int) -> tuple:
        """
        Creates a Notion toggle block.

        Args:
            lines (List[str]): The list of lines in the markdown content.
            index (int): The current index in the lines list.

        Returns:
            tuple: A tuple containing the Notion toggle block and the updated index.
        """
        index += 1  # Skip <details>
        summary = lines[index].replace('<summary>', '').replace('</summary>', '').strip()
        index += 1
        content = []
        
        while index < len(lines) and not lines[index].strip().startswith('</details>'):
            content.append(lines[index])
            index += 1
        
        return {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": summary}}],
                "children": self.markdown_to_blocks('\n'.join(content))
            }
        }, index

    def _create_paragraph_block(self, line: str) -> Dict:
        """
        Creates a Notion paragraph block.

        Args:
            line (str): The line containing the paragraph text.

        Returns:
            Dict: A Notion paragraph block.
        """
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": line.strip()}}]
            }
        }

if __name__ == "__main__":
    converter = NotionAgent()
    markdown_content = """
# Titre Principal

## Section 1

- Liste item 1
 - Sous-item
 - Sous-item 2
- Liste item 2

1. Premier point
 1. Sous-point
2. Second point

```python
print("Hello World!")
Une citation inspirante

Tâche non faite

Tâche faite

<details>
<summary>Cliquez pour voir plus</summary>
Contenu caché ici
</details>
"""
    page_id = converter.create_page_from_markdown(
    markdown_content,
    page_title="Nouvelle Page TESTE"
    )
    print(f"La page a été créée avec succès. ID de la page : {page_id}")