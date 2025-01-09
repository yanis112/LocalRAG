import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback
from markitdown import MarkItDown

class GoogleSheetsAgent:
    """
    Agent for interacting with Google Sheets API, saving sheets as temporary Excel files,
    and then converting them to Markdown files.
    
    Attributes:
        credentials_path (str): Path to Google Sheets credentials JSON file
        save_path (str): Directory path where Markdown files will be saved
        temp_path (str): Directory path where temporary Excel files will be saved
        scope (list): Google API scope permissions
        client (gspread.Client): Authorized Google Sheets client
    """

    def __init__(self, credentials_path: str, save_path: str = "./exports", temp_path: str = "./temp_exports"):
        """
        Initialize the GoogleSheetsAgent with credentials and save locations.

        Args:
            credentials_path (str): Path to the Google Sheets credentials JSON file
            save_path (str): Directory where Markdown files will be saved (default: ./exports)
            temp_path (str): Directory where temporary Excel files will be saved (default: ./temp_exports)
        """
        self.credentials_path = credentials_path
        self.save_path = save_path
        self.temp_path = temp_path
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive'
        ]
        self._authenticate()
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)

    def _authenticate(self):
        """Authenticate with Google Sheets API using service account credentials."""
        try:
            print(f"Authenticating using credentials from {self.credentials_path}")
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_path, 
                self.scope
            )
            self.client = gspread.authorize(credentials)
            print("Authentication successful")
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            traceback.print_exc()
            raise
        
    def fetch_and_save(self, spreadsheet_name: str):
        """
        Fetch all sheets from a spreadsheet, save them as temporary Excel files,
        convert them to Markdown, and then delete the temporary files.

        Args:
            spreadsheet_name (str): Name of the Google Spreadsheet
        """
        try:
            print(f"Opening spreadsheet: {spreadsheet_name}")
            spreadsheet = self.client.open(spreadsheet_name)
            
            for worksheet in spreadsheet.worksheets():
                try:
                    sheet_name = worksheet.title
                    print(f"Processing sheet: {sheet_name}")
                    
                    # Get all values including headers
                    values = worksheet.get_all_values()
                    
                    if not values:
                        print(f"Warning: Sheet '{sheet_name}' is empty")
                        continue
                    
                    # Create DataFrame with explicit headers from first row
                    headers = values[0]
                    data = values[1:]
                    df = pd.DataFrame(data, columns=headers)
                    
                    # Save as temporary Excel file
                    temp_xlsx_path = os.path.join(self.temp_path, f"{sheet_name}.xlsx")
                    df.to_excel(temp_xlsx_path, index=False)
                    print(f"Saved temporary Excel: {temp_xlsx_path}")

                    # Convert to Markdown
                    md = MarkItDown()
                    result = md.convert(temp_xlsx_path)
                    
                    # Save Markdown file
                    output_path = os.path.join(self.save_path, f"{sheet_name}.md")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)
                    print(f"Saved Markdown: {output_path}")
                    
                    # Delete temporary Excel file
                    os.remove(temp_xlsx_path)
                    print(f"Deleted temporary file: {temp_xlsx_path}")

                except Exception as e:
                    print(f"Error processing sheet '{sheet_name}': {str(e)}")
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Error accessing spreadsheet: {str(e)}")
            traceback.print_exc()
            raise
    
# Example usage:
if __name__ == "__main__":
    agent = GoogleSheetsAgent(
        credentials_path='google_json_key/python-sheets-446015-aa8eef72c872.json',
        save_path='data/sheets',
        temp_path='temp'
    )
    agent.fetch_and_save(spreadsheet_name='RechercheEmploi')