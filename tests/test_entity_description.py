from main_utils.generation_utils import entity_description   
import yaml 
if __name__ == '__main__':
    
    #load config file config/config.yaml
    with open('config/config.yaml') as file:
        config = yaml.safe_load(file)

    
    print(entity_description('Alice',default_config=config))