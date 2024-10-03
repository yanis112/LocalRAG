import os
import ssl
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from mattermostdriver import Driver, Websocket
import asyncio

class SSLAdapter(HTTPAdapter):
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

# Create an SSL context for the client-side connection
ssl_context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Set the SSL context for the requests library
session = requests.Session()
session.mount('https://', SSLAdapter(ssl_context=ssl_context))

# Driver configuration with the correct SSL context for the WebSocket connection
driver = Driver(
    {
        "url": str(os.getenv("URL_MATTERMOST")),
        "token": str(os.getenv("MATTERMOST_TOKEN")),
        "scheme": "https",
        "port": 443,
        "verify": False,  # Ensure SSL verification is disabled
        "session": session,
        "websocket_ssl_context": ssl_context,  # Pass the SSL context for WebSocket
    }
)

def get_all_teams():
    driver.login()
    teams_list = driver.teams.get_teams()

    page = 0
    per_page = 100
    public_channels_ids = []
    while True:
        try:
            print("scrapping page: ", page)
            channels_page = driver.teams.get_public_channels(
                teams_list[0]["id"], params={"page": page, "per_page": per_page}
            )
            if not channels_page:
                break
            public_channels_ids.extend(channels_page)
            page += 1
        except Exception:
            break

    public_channels = []
    for channel_id in public_channels_ids:
        try:
            channel = driver.channels.get_channel(channel_id["id"])
            public_channels.append(channel)
        except Exception:
            pass

    return public_channels

def handle_message(event):
    post = event.get("data", {}).get("post")
    if post:
        post = json.loads(post)
        if post["message"] == "hello":
            driver.posts.create_post(
                {"channel_id": post["channel_id"], "message": "Hi there!"}
            )

async def main(channel_name="Chatbot_Test"):
    driver.login()

    # list_public_channels = get_all_teams()
    # print("list_public_channels: ", list_public_channels)

    # channel = None
    # channel_id = None
    # corresponding_team_id = None
    # for ch in list_public_channels:
    #     if ch["display_name"] == channel_name:
    #         channel = ch
    #         channel_id = ch["id"]
    #         corresponding_team_id = ch["team_id"]
    #         break
    
    #print("trying to listen to channel: ", channel_name)
    #driver.init_websocket(event_handler=handle_message)
    driver.websocket = Websocket(
            driver.options, driver.client.token
        )
    print("trying to connect to websocket")
    await driver.websocket.connect(handle_message)

if __name__ == "__main__":
    asyncio.run(main())