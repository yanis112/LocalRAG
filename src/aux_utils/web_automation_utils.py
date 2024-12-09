from helium import *
import os
import json
from langchain_core.prompts import PromptTemplate

class WebAutomationAgent:
    def __init__(self, global_objective=None):
        self.vision_model_name = ''
        self.global_objective = global_objective
        self.action_memory = []  # action memory list all the textual actions taken at each steps
        self.screenshots_path = 'aux_data/screenshots/'
        # define possible actions and their descriptions
        self.action_dict = {
            'write': 'write text into a field',
            'click': 'click on a button or other clickable element',
            'press': 'press a key',
            'go_to': 'go to a specific url',
        }
        self.next_action = None

        from src.aux_utils.image_analysis import ImageAnalyzerAgent
        self.vision_agent = ImageAnalyzerAgent()
        # load the prompt from prompts/state2action.txt
        with open('prompts/state2action.txt', 'r') as f:
            self.state2action_template = PromptTemplate.from_template(f.read())

    def get_current_state(self):
        """Takes a screenshot of the current webpage"""
        # delete the current screenshot if it exists
        if os.path.exists(self.screenshots_path + 'screenshot.png'):
            os.remove(self.screenshots_path + 'screenshot.png')
        get_driver().save_screenshot(self.screenshots_path + 'screenshot.png')

    def get_action_for_state(self):
        """Use the language-vision model to produce the next action given the current state"""
        vllm_answer = self.vision_agent.describe_advanced(
            prompt=self.state2action_template.format(
                global_objective=self.global_objective,
                action_dict=str(self.action_dict)
            ),
            image_path=self.screenshots_path + 'screenshot.png'
        )
        print("Answer from the Vision-LLM model:", vllm_answer)
        #we have to transform the str dict back to a real dict
        self.next_action = json.loads(vllm_answer)

    def execute_action(self):
        """Execute the next action determined by the vision model"""
        if self.next_action:
            action = self.next_action.get('action')
            value = self.next_action.get('value')

            if action == 'write':
                write(value)
            elif action == 'click':
                click(value)
            elif action == 'press':
                press(value)
            elif action == 'go_to':
                go_to(value)
            else:
                print(f"Unknown action: {action}")

            # Add the action to the action memory
            self.action_memory.append(self.next_action)
            self.next_action = None

    def run(self):
        """Run the automation process"""
        start_chrome(headless=True)
        while True:
            self.get_current_state()
            self.get_action_for_state()
            self.execute_action()

# Example usage
agent = WebAutomationAgent(global_objective="Find the contact page")
agent.run()