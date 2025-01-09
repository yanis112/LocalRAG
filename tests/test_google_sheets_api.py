import gspread
from oauth2client.service_account import ServiceAccountCredentials
import sys
import traceback

def main():
    print("Starting Google Sheets API test...")
    
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive']
    
    try:
        print("Reading credentials...")
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            'google_json_key/python-sheets-446015-aa8eef72c872.json',
            scope
        )
        
        gc = gspread.authorize(credentials)
        print("Authorizing with Google Sheets...")
        
        spreadsheet = gc.open('RechercheEmploi')
        print("Opening spreadsheet 'RechercheEmploi'...")
        
        sheet_info = spreadsheet.sheet1
        print("Accessing first worksheet...")
        
        # Get and print column headers
        headers = sheet_info.row_values(1)  # Get first row (headers)
        print("Column headers:")
        print(headers)
            
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()