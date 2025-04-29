from openai import OpenAI
import base64
client = OpenAI()
prompt = """
make a logo for a product called guac that has guac in dark green 
and then in the "a" of the word is a bowl of guacamole. 
"""
result = client.images.edit(
    model="gpt-image-1",
    image= open("guac_shell.png", "rb"),
    prompt=prompt
)
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)
# Save the image to a file
with open("guac_logo_edit.png", "wb") as f:
    f.write(image_bytes)