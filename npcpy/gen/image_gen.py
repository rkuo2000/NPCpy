########
########
########
########
########
########
######## IMAGE GENERATION
########

import os


from litellm import image_generation
from npcpy.npc_sysenv import (
    NPCSH_IMAGE_GEN_MODEL,
    NPCSH_IMAGE_GEN_PROVIDER,
)
import PIL
import io


def generate_image_diffusers(
    prompt: str,
    model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cpu",
    height: int = 256,
    width: int = 256,
):
    """
    Function Description:
        This function generates an image using the Stable Diffusion API.
    Args:
        prompt (str): The prompt for generating the image.
        model_id (str): The Hugging Face model ID to use for Stable Diffusion.
        device (str): The device to run the model on ('cpu' or 'cuda').
    Returns:
        PIL.Image: The generated image.
    """
    # Load the Stable Diffusion pipeline
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt, height=height, width=width)
    image = image.images[0]
    # ["sample"][0]
    image.show()

    return image

def gpt_image_gen(
    prompt: str,
    model: str = "gpt-image-1",
    attachments: list = None,
    height: int = 256,
    width: int = 256,
    n_images: int = 1,
):
    """
    Function Description:
        This function generates an image using the OpenAI API.
    Args:
        prompt (str): The prompt for generating the image.
        model (str): The model to use for generating the image.
        api_key (str): The API key  .
        api_url (str): The API URL  LITELLM DOES NOT YET SUPPORT CUSTOM.
    Keyword Args:
        None
    Returns:
        str: The URL of the generated image.
    """
    print(height, width, model, prompt)
    from openai import OpenAI
    import base64
    client = OpenAI()
    if attachments is not None:
        result = client.images.edit(
            model=model,
            image= attachments[0] if len(attachments)==1 else attachments,
            prompt=prompt, 
            n=n_images,
            size=f"{height}x{width}",
        )
        
    else:
        result = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=f"{height}x{width}",
        )        

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    #return a PIL image 
    image = PIL.Image.open(io.BytesIO(image_bytes))
    # Save the image to a file
    return image

def generate_image_litellm(
    prompt: str,
    model: str = NPCSH_IMAGE_GEN_MODEL,
    provider: str = NPCSH_IMAGE_GEN_PROVIDER,
    height: int = 256,
    width: int = 256,
    attachments: list = None,
    n_images: int = 1,
    api_key: str = None,
    api_url: str = None,
    npc=None,
) -> str:
    """
    Function Description:
        This function generates an image using the OpenAI API.
    Args:
        prompt (str): The prompt for generating the image.
        model (str): The model to use for generating the image.
        api_key (str): The API key  .
        api_url (str): The API URL  LITELLM DOES NOT YET SUPPORT CUSTOM.
    Keyword Args:
        None
    Returns:
        str: The URL of the generated image.
    """
    if model is None:
        model = "runwayml/stable-diffusion-v1-5"
    if provider == "openai":
        if model == 'gpt-image-1':
            return gpt_image_gen(prompt, 
                                 model=model, 
                                 attachments=attachments, 
                                 height=height, 
                                 width=width)
        
        elif "dall" not in model:
            model = "dall-e-2"
    if provider == "diffusers":
        return generate_image_diffusers(prompt, model, height=height, width=width)
    else:

        if f"{height}x{width}" not in [
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1792",
            "1792x1024",
        ]:
            raise ValueError(
                f"Invalid image size: {height}x{width}. Please use one of the following: 256x256, 512x512, 1024x1024, 1024x1792, 1792x1024"
            )
        #print(model, provider)
        return image_generation(
            prompt=prompt,
            model=f"{provider}/{model}",
            n=n_images,
            size=f"{height}x{width}",
            api_key=api_key,
            api_base=api_url,
        )
