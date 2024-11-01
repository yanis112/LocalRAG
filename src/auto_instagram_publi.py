from src.generation_utils_v2 import LLM_answer_v3
import yaml

class InstagramDescriptor:
    def __init__(self):
        self.ex1= """🚀 The Empire's Inquisitors Rise 🔥 The Holy Galactic Byzantine Empire deploys its most fearsome droid Inquisitor, Saint Aurelio di San Giovanni, to lead the charge against the vile MiVfe—demonic alien entities threatening the empire. As the battle intensifies beneath a golden sun, his unyielding faith and lethal precision carve through the enemy, paving the way for the empire's triumph over alien darkness. 🌞⚫ Join the sacred war and witness the dawning of a new era! ✨
#AIArt #EpicHistory #HolyEmpire #GalacticByzantine #Inquisitors #DemonicAliens #SciFiBattle #FuturisticFaith #SpaceOdyssey #FantasyArt #TechAndTradition #EpicSaga #AlienInvasion #CosmicWar #AIArtist #ArtOfInstagram #midjourney #stablediffusionartwork #ai #aiartcommunity
3 j"""
        self.ex2=""" 🔥🌑 Witness the epic scenes from *The Silmarillion* brought to life in stunning AI visuals! ⚔️ The mighty Balrogs blaze with fury, while the Elves stand in defiance 🏹✨. Cities crumble under the shadow of the Dark Lord, Morgoth himself, as the fall of kingdoms echoes through time 🏰💀. A legendary tale of light and darkness, where heroes rise and empires fall.
        #AIArt #Silmarillion #Tolkien #EpicFantasy #Morgoth #Balrog #Elves #FallOfCities #FantasyWorlds #DigitalArt #AIStories #MythicLegends #middleearth
        # """
        self.ex3="""One Piece imagined in Ghibli Studio style ! 🔥#onepiece #ghibli #ghiblistudio #ai #aiart #aiartcommunity #midjourney #abstractart #art #onepiecefan #onepieceanime"""
        self.ex4="""👽👽 Alien creature reported in the countryside near Nashville, these creatures reportedly left no survivor. #ai #aiart #aiartcommunity #midjourney #midjourneyart #aliens #art #artist"""
        self.config= yaml.safe_load(open("config/config.yaml"))
            
    def generate_description(self,base_prompt):
        """Takes a prompt containing a description of an image as input and generates a complete prompt with instruction on how to generate a instagram post description for the image
        

        Args:
            base_prompt (_type_): _description_
        """
        
        image_description=base_prompt
        
        prompt= f"""Here is the description of an image i will post on instagram. I need you to write an instagram description for the post,
        with all the professional elements that are needed to make it engaging and attractive to the audience, such as several hashtags (at least ten), emojis, lore elements, a short story to explain image context, etc..., Here is the
        image description: ### {image_description}. ### """ 
        
        exemples_prompt= f"""Here are some examples of good quality instagram descriptions to help you: Example 1: {self.ex1} \n\n Example 2: {self.ex2} \n\n Example 3: {self.ex3} \n\n Example 4: {self.ex4} \n\n """
        
        
        final_prompt_part=""" Now, please write a description fitting the image description that is engaging and attractive to the audience, with at least ten hashtags (#ai #midjourney #midjourneyart #aiart are mandatory !), emojis, lore elements, a short story to explain image context, etc..., anwser only the description and without preamble. """
        
        final_prompt= prompt + exemples_prompt + final_prompt_part
        
        print("FINAL PROMPT: ", final_prompt)
        
        answer=LLM_answer_v3(final_prompt,model_name=self.config["model_name"],llm_provider=self.config["llm_provider"], stream=False)
        
        return answer




