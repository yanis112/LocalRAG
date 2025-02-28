import json
import os
import time
from PIL import Image
from helium import click, get_driver, go_to, press, start_chrome, write
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

from src.main_utils.generation_utils_v2 import LLM_answer_v3
from src.main_utils.agentic_rag_utils import QueryBreaker
from subprojects.omni_parser_agent.omniparser_agent import OmniParserAgent
from langchain.tools import tool


class WebAutomationAgentV2:
    def __init__(self, global_objective=None,config=None):
        self.global_objective = global_objective
        self.config = config
        self.model_name=config["model_name"]
        self.llm_provider=config["llm_provider"]
        self.temperature=config["temperature"]
        self.action_memory = ["No actions taken yet"]
        self.screenshots_path = os.path.join("aux_data", "screenshots")
        self.action_dict = config["allowed_actions"]
        
        MODEL_PATH = "c:/Users/Yanis/Documents/RAG/omniparser_v2/icon_detect/model.pt"
        self.omni_parser_agent = OmniParserAgent(MODEL_PATH)

        # Initialize query breaker with same configuration
        self.query_breaker = QueryBreaker(self.config)

        # Load prompt template from file if exists, else use default
        self.prompt_template_path = os.path.join("prompts", "state2action_v2.txt")
        if os.path.exists(self.prompt_template_path):
            with open(self.prompt_template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
        else:
            template_str = (
                "Global Objective: {global_objective}\n"
                "OmniParser Output: {omni_parser_output}\n"
                "Previous Actions: {previous_actions}\n"
                "Available Actions: {action_dict}\n"
                "Based on the above, determine the next appropriate tool call in JSON format."
            )
        self.state2action_template = PromptTemplate.from_template(template_str)
        
        # Define tools with proper type hints
        @tool(return_direct=True)
        def write_text(text: str, into: str) -> Dict[str, Any]:
            """Write text into a specified field on the website page"""
            write(text, into=into)
            return {"status": "success", "action": f"Wrote '{text}' into '{into}'"}

        @tool(return_direct=True)
        def click_element(element: str) -> Dict[str, Any]:
            """Click on a specified element on the website page (button, link, etc.)"""
            click(element=element)
            return {"status": "success", "action": f"Clicked '{element}'"}

        @tool(return_direct=True)
        def press_key(key: str) -> Dict[str, Any]:
            """Press a specified key on the keyboard"""
            press(key=key)
            return {"status": "success", "action": f"Pressed key '{key}'"}

        @tool(return_direct=True)
        def go_to_url(url: str) -> Dict[str, Any]:
            """Navigate to a specified URL"""
            go_to(url=url)
            time.sleep(2)  # Add delay to ensure page loads before screenshot
            return {"status": "success", "action": f"Navigated to '{url}'"}

        # Store tools as instance methods
        self.tools = [write_text, click_element, press_key, go_to_url]

    def get_current_state(self):
        # Ensure screenshots directory exists
        if not os.path.exists(self.screenshots_path):
            os.makedirs(self.screenshots_path)
        screenshot_path = os.path.join(self.screenshots_path, "screenshot.png")
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)
        time.sleep(1)  # Add small delay to ensure page is stable
        driver = get_driver()
        driver.save_screenshot(screenshot_path)
        return screenshot_path

    def get_omniparser_output(self, screenshot_path: str) -> str:
        """Retrieves and formats the output from the OmniParserAgent for a given screenshot."""
        try:
            img = Image.open(screenshot_path)
            _, label_coordinates, boxes_with_content = self.omni_parser_agent.process_image(img)
            
            # Check if no elements were detected
            if not label_coordinates or not boxes_with_content:
                print("No UI elements detected in the current page screenshot")
                return "No visualization of the page yet - no UI elements detected"
                
            formatted_output = self.omni_parser_agent.format_output_for_llm(label_coordinates, boxes_with_content)
            return formatted_output
        except Exception as e:
            print(f"Error obtaining OmniParser output: {e}")
            return "No visualization of the page yet - error occurred during processing"

    def get_and_execute_action_for_state(self, omni_parser_output: str, current_subtask: str) -> dict:
        """Determines and executes the next action based on current state and subtask"""
        prompt = self.state2action_template.format(
            global_objective=current_subtask,  # Use current subtask instead of global objective
            omni_parser_output=omni_parser_output,
            previous_actions=json.dumps(self.action_memory),
            action_dict=json.dumps(self.action_dict)
        )
        
        content, tool_calls = LLM_answer_v3(
            prompt=prompt,
            model_name=self.model_name,
            llm_provider=self.llm_provider,
            temperature=self.temperature,
            stream=False,
            tool_list=self.tools,
        )
        
        print(f"LLM Response for sub-task: {content}")
        print(f"Tool calls for sub-task: {tool_calls}")
                
        # Execute the tool calls for this sub-task
        for tool_call in tool_calls:
            print(f"Executing tool call: {tool_call}")
            try:
                tool_func = next(t for t in self.tools if t.name == tool_call['name'])
                args = tool_call['args']
                result = tool_func.invoke(args)
                print(f"Result: {result}")
                # Add executed action to memory
                self.action_memory.append(tool_call)
                # Take screenshot after each action
                self.get_current_state()
            except Exception as e:
                print(f"Error executing tool call: {str(e)}")
                return False
        return True

    def run(self):
        """Run the automation process with query breaking"""
        if not self.global_objective:
            raise ValueError("Global objective must be set before running automation")
            
        # Break down the global objective into sub-tasks
        print("Breaking down objective into sub-tasks...")
        sub_tasks = self.query_breaker.break_query(
            self.global_objective,
            context="Web automation tasks for browser interactions",
            unitary_actions=list(self.action_dict.keys())
        )
        print(f"Sub-tasks identified: {sub_tasks}")

        # Execute each sub-task sequentially
        for task_index, sub_task in enumerate(sub_tasks):
            print(f"\nProcessing sub-task: {sub_task}")
            
            # Get initial state without taking screenshot yet
            if task_index>0:
                screenshot_path = self.get_current_state()
                omni_parser_output = self.get_omniparser_output(screenshot_path)
                print(f"OmniParser output:\n{omni_parser_output}")
                
            else: #we have to first execute first task before trying to get a state
                omni_parser_output= "No page visualization requiered fot his task"
            
            # Execute action for current sub-task
            success = self.get_and_execute_action_for_state(omni_parser_output, sub_task)
            if not success:
                print(f"Failed to execute sub-task: {sub_task}")
                continue
                
            time.sleep(1)  # Small delay between sub-tasks


if __name__ == "__main__":
    import yaml
    with open("config/omniparser_agent_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    start_chrome(headless=False)
    agent = WebAutomationAgentV2(
        global_objective="Go to https://cas.centrale-med.fr/login?service=https%3A%2F%2Fmoodle.centrale-med.fr%2Flogin%2Findex.php and login with password 'test2' and username 'test1'",
        config=config
    )
    agent.run()
