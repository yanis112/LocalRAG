from instagrapi import Client

def post_to_instagram(username, password, image_path, caption):
    cl = Client()
    cl.login(username=username, password=password)
    
    # Upload the photo
    media = cl.photo_upload(
        path=image_path,
        caption=caption
    )
    
    cl.logout()
    return media

# Usage
post_to_instagram("aiandcivilization", "labeyrie06", "received_image.png", "Check out this cool photo! #instagram #python")