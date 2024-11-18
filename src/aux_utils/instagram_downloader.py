import instaloader

# Create an instance of Instaloader
L = instaloader.Instaloader()

# Specify the username of the profile you want to download from
username = "ella.hunadi"

try:
    # Load the profile
    profile = instaloader.Profile.from_username(L.context, username)
    
    # Download all posts
    for post in profile.get_posts():
        L.download_post(post, target=username)
    
    print(f"All posts from {username} have been downloaded successfully.")

except instaloader.exceptions.ProfileNotExistsException:
    print(f"The profile {username} does not exist.")
except instaloader.exceptions.PrivateProfileNotFollowedException:
    print(f"The profile {username} is private and you're not following it.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
