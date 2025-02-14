import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback
from src.main_utils.generation_utils_v2 import LLM_answer_v3
from src.main_utils.agentic_rag_utils import QueryBreaker
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

class GoogleSheetsAgent:
    """
    Agent for interacting with Google Sheets API with natural language capabilities.
    """

    def __init__(self, credentials_path: str, save_path: str = "./exports", temp_path: str = "./temp_exports",
                 model_name: str = "gpt-4", llm_provider: str = "openai"):
        """Initializes the agent with Google Sheets credentials and LLM configuration.

        Args:
            credentials_path (str): Path to Google Sheets API credentials
            save_path (str): Path to save exported files
            temp_path (str): Path for temporary files
            model_name (str): Name of the LLM model to use
            llm_provider (str): Provider of the LLM (e.g., 'openai', 'anthropic')
        """
        self.credentials_path = credentials_path
        self.save_path = save_path
        self.temp_path = temp_path
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive'
        ]
        self._authenticate()
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)
        # Initialize query breaker with same configuration
        self.config = {
            "model_name": model_name,
            "llm_provider": llm_provider,
            "prompt_language": "en",  # Using English for consistency
            "temperature": 0.7
        }
        self.query_breaker = QueryBreaker(self.config)

    def _authenticate(self):
        """Authenticates with Google Sheets API."""
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

    def _format_table_with_indices(self, values: list) -> str:
        """Format table content with cell indices for better LLM understanding.
        
        Each cell will be represented as: {content} [R{{row}},C{{col}}]
        
        Args:
            values (list): List of lists containing the worksheet values
            
        Returns:
            str: Markdown formatted table with indices
        """
        if not values:
            return ""
            
        result = []
        # Create header row
        header = "| "
        separator = "|"
        for col_idx, cell in enumerate(values[0], start=1):
            header += f"{cell} [R1,C{col_idx}] | "
            separator += " --- |"
        result.append(header.strip())
        result.append(separator)
        
        # Create data rows
        for row_idx, row in enumerate(values[1:], start=2):
            row_str = "| "
            for col_idx, cell in enumerate(row, start=1):
                row_str += f"{cell} [R{row_idx},C{col_idx}] | "
            result.append(row_str.strip())
            
        return "\n".join(result)

    def fetch_and_save(self, spreadsheet_name: str):
        """Fetches and saves sheets as Markdown."""
        try:
            print(f"Opening spreadsheet: {spreadsheet_name}")
            spreadsheet = self.client.open(spreadsheet_name)

            for worksheet in spreadsheet.worksheets():
                try:
                    sheet_name = worksheet.title
                    print(f"Processing sheet: {sheet_name}")
                    values = worksheet.get_all_values()

                    if not values:
                        print(f"Warning: Sheet '{sheet_name}' is empty")
                        continue

                    # Use the new formatting function instead of pandas conversion
                    formatted_table = self._format_table_with_indices(values)
                    output_path = os.path.join(self.save_path, f"{sheet_name}.md")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(formatted_table)
                    print(f"Saved Markdown: {output_path}")

                except Exception as e:
                    print(f"Error processing sheet '{sheet_name}': {str(e)}")
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Error accessing spreadsheet: {str(e)}")
            traceback.print_exc()
            raise

    def list_sheet_names(self, spreadsheet_name: str) -> list:
        """Lists sheet names in a spreadsheet."""
        try:
            spreadsheet = self.client.open(spreadsheet_name)
            sheet_names = [worksheet.title for worksheet in spreadsheet.worksheets()]
            print(f"Sheet names in '{spreadsheet_name}': {sheet_names}")
            return sheet_names
        except Exception as e:
            print(f"Error listing sheet names: {str(e)}")
            traceback.print_exc()
            raise
    @staticmethod
    @tool(return_direct=True)
    def add_row(spreadsheet_name: str, sheet_name: str, row_data: list[str]) -> str:
        """Adds a row to a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            row_data (list[str]): List of values to add as a new row.

        Returns:
            str: Success message indicating the row was added.
        """
        try:
            agent = GoogleSheetsAgent(
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            values = worksheet.get_all_values()
            next_row = len(values) + 1 if values else 1
            range_to_update = f'A{next_row}'
            worksheet.update(range_to_update, [row_data], value_input_option='USER_ENTERED')
            return f"Row added successfully at position {next_row}"
        except Exception as e:
            return f"Error adding row: {str(e)}"

    @staticmethod
    @tool(return_direct=True)
    def add_column(spreadsheet_name: str, sheet_name: str, column_data: list[str], column_index: int | None = None) -> str:
        """Adds a column to a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            column_data (list[str]): List of values for the new column.
            column_index (int, optional): Position to insert the column.

        Returns:
            str: Success message indicating the column was added.
        """
        try:
            agent = GoogleSheetsAgent(
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            if column_index is None:
                column_index = worksheet.col_count + 1
            num_rows = worksheet.row_count
            if len(column_data) < num_rows:
                column_data.extend([''] * (num_rows - len(column_data)))
            values = [[value] for value in column_data]
            worksheet.insert_cols(values, col=column_index)
            return f"Column added successfully at position {column_index}"
        except Exception as e:
            return f"Error adding column: {str(e)}"

    @staticmethod
    @tool(return_direct=True)
    def modify_cell(spreadsheet_name: str, sheet_name: str, row: int, col: int, new_value: str) -> str:
        """Modifies a cell's value in a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            row (int): Row number (1-based).
            col (int): Column number (1-based).
            new_value (str): New value for the cell.

        Returns:
            str: Success message indicating the cell was updated.
        """
        try:
            # --- The ONLY changed part is within this try block ---
            agent = GoogleSheetsAgent(  # Keep original instantiation
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            col_letter = chr(ord('A') + col - 1)  # Keep A1 notation logic
            cell_range = f'{col_letter}{row}'
            worksheet.update(range_name=cell_range, values=[[new_value]], value_input_option='USER_ENTERED') # Corrected update call
            return f"Cell ({row}, {col}) updated successfully"
            # --- End of changed part ---
        except Exception as e:
            return f"Error modifying cell: {str(e)}"

    @staticmethod
    @tool(return_direct=True)
    def delete_row(spreadsheet_name: str, sheet_name: str, row_index: int) -> str:
        """Deletes a row from a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            row_index (int): Index of row to delete (1-based).

        Returns:
            str: Success message indicating the row was deleted.
        """
        try:
            agent = GoogleSheetsAgent(
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            worksheet.delete_rows(row_index)
            return f"Row {row_index} deleted successfully"
        except Exception as e:
            return f"Error deleting row: {str(e)}"

    @staticmethod
    @tool(return_direct=True)
    def delete_column(spreadsheet_name: str, sheet_name: str, col_index: int) -> str:
        """Deletes a column from a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            col_index (int): Index of column to delete (1-based).

        Returns:
            str: Success message indicating the column was deleted.
        """
        try:
            agent = GoogleSheetsAgent(
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            worksheet.delete_columns(col_index)
            return f"Column {col_index} deleted successfully"
        except Exception as e:
            return f"Error deleting column: {str(e)}"

    @staticmethod
    @tool(return_direct=True)
    def clear_cell(spreadsheet_name: str, sheet_name: str, row: int, col: int) -> str:
        """Clears a cell's content in a Google Sheet.

        Args:
            spreadsheet_name (str): Name of the spreadsheet.
            sheet_name (str): Name of the sheet.
            row (int): Row number (1-based).
            col (int): Column number (1-based).

        Returns:
            str: Success message indicating the cell was cleared.
        """
        try:
            agent = GoogleSheetsAgent(
                credentials_path='google_json_key/python-sheets-key.json',
                save_path='data/sheets',
                temp_path='temp',
                model_name="gemini-2.0-pro-exp-02-05",
                llm_provider='google'
            )
            spreadsheet = agent.client.open(spreadsheet_name)
            worksheet = spreadsheet.worksheet(sheet_name)
            worksheet.update_cell(row, col, '')
            return f"Cell ({row, col}) cleared successfully"
        except Exception as e:
            return f"Error clearing cell: {str(e)}"
    def act(self, query: str, spreadsheet_name: str) -> bool:
        """Process a natural language query to perform actions on a Google Sheet.
        Now includes query breaking and sequential execution of sub-tasks.
        
        Args:
            query (str): Natural language query describing the action to perform
            spreadsheet_name (str): Name of the spreadsheet to operate on
            
        Returns:
            bool: True if the action was successful, False otherwise
        """
        try:
            # Get available tools
            tools = [
                GoogleSheetsAgent.add_row,
                GoogleSheetsAgent.add_column,
                GoogleSheetsAgent.modify_cell,
                GoogleSheetsAgent.delete_row,
                GoogleSheetsAgent.delete_column,
                GoogleSheetsAgent.clear_cell
            ]
            # Extract tool names for the query breaker
            tool_names = [tool.name for tool in tools]
            
            # First get the sheet names
            sheet_names = self.list_sheet_names(spreadsheet_name)
            
            # Fetch and save the current state of sheets
            self.fetch_and_save(spreadsheet_name)
            
            # Read the markdown content of the sheets
            table_content = ""
            for sheet_name in sheet_names:
                md_path = os.path.join(self.save_path, f"{sheet_name}.md")
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        table_content += f"# {sheet_name}\n{f.read()}\n\n"

            # Step 1: Break down the query into sub-tasks using QueryBreaker
            print("Breaking down query into sub-tasks...")
            print("TOOL NAMES:", tool_names)
            sub_tasks = self.query_breaker.break_query(query, context=table_content, unitary_actions=tool_names)
            print(f"Sub-tasks identified: {sub_tasks}")

            # Load and format the prompt template for tool selection
            with open("prompts/google_sheets_agent_prompt.txt", "r", encoding='utf-8') as f:
                template = f.read()
            
            prompt_template = PromptTemplate.from_template(template)

            # Step 2: Process each sub-task sequentially
            overall_success = True
            import time
            for sub_task in tqdm(sub_tasks, desc="Processing sub-tasks"):
                time.sleep(1)  # Add a delay to avoid rate limiting
                print(f"\nProcessing sub-task: {sub_task}")
                
                # Format prompt for this specific sub-task
                formatted_prompt = prompt_template.format(
                    spreadsheet_name=spreadsheet_name,
                    sheet_names=sheet_names,
                    table_content=table_content,
                    query=sub_task  # Using sub-task instead of original query
                )

                # Get tool calls for this sub-task
                content, tool_calls = LLM_answer_v3(
                    prompt=formatted_prompt,
                    model_name=self.model_name,
                    llm_provider=self.llm_provider,
                    tool_list=tools
                )

                print(f"LLM Response for sub-task: {content}")
                print(f"Tool calls for sub-task: {tool_calls}")
                
                # Execute the tool calls for this sub-task
                for tool_call in tool_calls:
                    print(f"Executing tool call: {tool_call}")
                    try:
                        tool_func = next(t for t in tools if t.name == tool_call['name'])
                        args = tool_call['args']
                        result = tool_func.invoke(args)
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Error executing tool call: {str(e)}")
                        overall_success = False

            return overall_success

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()
            return False

# Example usage:
if __name__ == "__main__":
    agent = GoogleSheetsAgent(
        credentials_path='google_json_key/python-sheets-key.json',
        save_path='data/sheets',
        temp_path='temp',
        model_name= "gemini-2.0-flash", #"deepseek-r1-distill-llama-70b", # "deepseek-r1-distill-llama-70b", #, "gemini-2.0-flash", #"deepseek-r1-distill-llama-70b",
        llm_provider='google'
        #"google" #'groq'
    )
    
    # Test the act method with a natural language query
    spreadsheet_name = "test_google_sheet"
    query = "Add a new row to the table with values ['Company ABC', 'Software Engineer', '2024-03-15', 'Applied'] to the first sheet, then modify the applied status for Thales to not applied"
    #query = "add two random website names, one for Thales one for Boeing"

    
    print(f"Processing query: {query}")
    success = agent.act(query, spreadsheet_name)
    print(f"Operation {'succeeded' if success else 'failed'}")