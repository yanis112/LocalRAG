import json
import os
import re

from helium import click, get_driver, go_to, press, start_chrome, write
from langchain_core.prompts import PromptTemplate


class WebAutomationAgent:
    def __init__(self, global_objective=None):
        self.vision_model_name = ""
        self.global_objective = global_objective
        self.action_memory = [
            "No actions taken yet"
        ]  # action memory list all the textual actions taken at each steps
        self.screenshots_path = "aux_data/screenshots/"
        self.vllm_name = "gemini-2.0-flash-exp"
        self.vllm_provider = "gemini"
        # define possible actions and their descriptions
        self.action_dict = {
            "write": "write text into a field e.g: write('username', into='Username'), the 'into' argument must correspond to the name of the field (written on the label)",
            "click": "click on a button or other clickable element e.g: click(element='Login'), the 'element' argument must correspond to the name of the button",
            "press": "press a key e.g: press(key=ENTER)",
            "go_to": "go to a specific url e.g: go_to(url='https://www.google.com')",
        }
        self.next_action = None

        from src.aux_utils.vision_utils_v2 import ImageAnalyzerAgent

        self.vision_agent = ImageAnalyzerAgent()
        # load the prompt from prompts/state2action.txt
        with open("prompts/state2action.txt", "r") as f:
            self.state2action_template = PromptTemplate.from_template(f.read())

    def get_current_state(self):
        """Takes a screenshot of the current webpage"""
        # create the screenshots directory if it does not exist
        if not os.path.exists(self.screenshots_path):
            print("Screenshots directory does not exist, creating it...")
            os.makedirs(self.screenshots_path)
        # delete the current screenshot if it exists
        if os.path.exists(self.screenshots_path + "screenshot.png"):
            os.remove(self.screenshots_path + "screenshot.png")
        get_driver().save_screenshot(self.screenshots_path + "screenshot.png")

    def get_action_for_state(self):
        """Use the language-vision model to produce the next action given the current state"""
        vllm_answer = self.vision_agent.describe(
            prompt=self.state2action_template.format(
                global_objective=self.global_objective,
                action_dict=json.dumps(self.action_dict),
                previous_actions=json.dumps(self.action_memory),
            ),
            image_path=self.screenshots_path + "screenshot.png",
            vllm_name=self.vllm_name,
            vllm_provider=self.vllm_provider,
        )
        # remove potential noise from vllm answer "```json" or "```"
        vllm_answer = re.sub(r"```json", "", vllm_answer)
        vllm_answer = re.sub(r"```", "", vllm_answer)

        print("Answer from the Vision-LLM model:", vllm_answer)
        # transform the str dict into a real dict object
        try:
            self.next_action = json.loads(vllm_answer)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            self.next_action = None

        # the answer is a str object representing the function call and its arguments e.g: "write(text='username', into='Username')"

    def execute_action(self):
        """Execute the next action determined by the vision model"""
        if self.next_action:
            # extract the name of the function from dict
            action_type = self.next_action["function"]["name"]
            # add the str of the next action to the action memory
            self.action_memory.append(json.dumps(self.next_action))

            if action_type == "write":
                # do helium write action
                write(
                    text=self.next_action["function"]["arguments"]["text"],
                    into=self.next_action["function"]["arguments"]["into"],
                )
            elif action_type == "click":
                click(element=self.next_action["function"]["arguments"]["element"])
            elif action_type == "press":
                press(key=self.next_action["function"]["arguments"]["key"])
            elif action_type == "go_to":
                go_to(url=self.next_action["function"]["arguments"]["url"])
            else:
                print(f"Unknown action: {action_type}")

            # Add the action to the action memory
            self.action_memory.append(self.next_action)
            self.next_action = None

    def run(self):
        """Run the automation process"""
        # start_chrome(headless=True)
        while True:
            self.get_current_state()
            self.get_action_for_state()
            self.execute_action()


# Example usage
if __name__ == "__main__":
    start_chrome(
        url="https://auth.centrale-marseille.fr/cas/login?service=https%3A%2F%2Fwmail.centrale-marseille.fr%2F"
    )
    agent = WebAutomationAgent(
        global_objective="Login to the Centrale Marseille email service"
    )
    agent.run()
