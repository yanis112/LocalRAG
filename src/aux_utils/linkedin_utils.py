from dotenv import load_dotenv
from linkedin_api import Linkedin
import os
from typing import List, Dict, Optional

class LinkedInScraper:
    def __init__(self):
        """Initialize LinkedIn scraper with credentials from .env file"""
        load_dotenv()
        self.username = os.getenv('LINKEDIN_USERNAME')
        self.password = os.getenv('LINKEDIN_PASSWORD')
        
        if not self.username or not self.password:
            raise ValueError("LinkedIn credentials not found in .env file")
        
        try:
            self.api = Linkedin(self.username, self.password)
        except Exception as e:
            raise ConnectionError(f"Failed to authenticate with LinkedIn: {str(e)}")

    def get_profile_posts(self, public_id: str, post_count: int = 10) -> Optional[List[Dict]]:
        """
        Fetch posts from a LinkedIn profile
        
        Args:
            public_id (str): LinkedIn public ID of the user
            post_count (int): Number of posts to retrieve (default: 10)
            
        Returns:
            List[Dict]: List of posts or None if failed
        """
        try:
            posts = self.api.get_profile_posts(public_id=public_id, post_count=post_count)
            return posts
        except Exception as e:
            print(f"Error fetching posts for {public_id}: {str(e)}")
            return None
        
        
if __name__ == "__main__":
    # Test LinkedInScraper class
    linkedin_scraper = LinkedInScraper()
    posts = linkedin_scraper.get_profile_posts(public_id="saurav-pawar-058143196")
    if posts:
        print(posts)
    else:
        print("Failed to fetch posts")