from suno import Suno, ModelVersions

#github links: https://github.com/Malith-Rukshan/Suno-API

client = Suno(
  cookie='__client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8ybXZROFV2Ymo5Z1pSNWV2YUZHWWJkOTAyUXMiLCJyb3RhdGluZ190b2tlbiI6Imk0aWIydGMxNDh3NWxxdGdlaXUwZ25oMnp2amZ0anUyZ2w5NjhmcWkifQ.oQHGpyjwv5uF9ELWGNmhYEmsnl_m5oNWASup5DXt6SMinl3bqYXcdyLwT3mO7GMXNR73XS5Q7xTmdZRaGhlPlnDVMLIDf2iepGx5uFEzi5A8Opgxmfy2EiWIuCWC6C7vQneAdzrPjfswL6QqDqW9JwyD5b48R9faBq39kjtoJ5tfoHtwjv88BZqN6P-TOxdI_p_R1vPBiI4OGuJxKJ1GBZp4ILpK4LUO-pgvyX1CDppc4LDL4ckLA3URrT9DrOZWrEGV3trgWdZB3NzcJiIKTGYCQwxMk05ZOt9d5IFNDDOonDPAso017GZKAvspUGBVZBFgI6HX5YYx9424J9vaTQ',
  model_version=ModelVersions.CHIRP_V3_5)

# Generate a song
songs = client.generate(prompt="A serene landscape", is_custom=False, wait_audio=True)

# Download generated songs
for song in songs:
    file_path = client.download(song=song)
    print(f"Song downloaded to: {file_path}")