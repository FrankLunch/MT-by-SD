!pip install diffusers --upgrade
!pip install invisible_watermark transformers accelerate safetensors
!pip install diffusers transformers accelerate torch torchvision

from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
from huggingface_hub import login

# Log in with your Hugging Face token
# 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
login("Replace with your actual token")  # Replace with your actual token

# Load the pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl",
    use_auth_token=True,  # Use token for authentication
    torch_dtype=torch.float16
)
pipe.to("cuda")

import random
import string
#lane same , slightly change the env
def MR1(image_path):
  # Load your input image
  input_image_path = image_path  # Replace with your image path
  input_image = Image.open(input_image_path).convert("RGB")
  input_image = input_image.resize((1024, 512))  # High resolution for better quality

  # Generate the new image
  prompt = "A realistic car running on the same lane, slightly changed background, cinematic high-quality lighting"
  negative_prompt = "low quality, distorted, cartoonish, unrealistic"
  output = pipe(
      prompt=prompt,
      image=input_image,
      strength=0.3,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]
  file_name = "generated_image_MR1+"+str(random.randint(10000, 99999))+".png"
  output.save(file_name)
  print("Generated image saved as"+file_name)

#change the weather
def MR2(image_path):
  # Load your input image
  input_image_path = image_path  # Replace with your image path
  input_image = Image.open(input_image_path).convert("RGB")
  input_image = input_image.resize((1024, 512))  # High resolution for better quality

  # Generate the new image
  prompt = "A realistic car running on the same lane, slightly changed the weather to heavy snow, cinematic high-quality lighting"
  negative_prompt = "low quality, distorted, cartoonish, unrealistic"
  output = pipe(
      prompt=prompt,
      image=input_image,
      strength=0.5,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]

  file_name = "generated_image_MR2+"+str(random.randint(10000, 99999))+".png"
  output.save(file_name)
  output.show()
  print("Generated image saved as"+file_name)

#Reversing the direction of the lane.
def MR3(image_path):
  # Load your input image
  input_image_path = image_path  # Replace with your image path
  input_image = Image.open(input_image_path).convert("RGB")
  input_image = input_image.resize((1024, 512))  # High resolution for better quality

  # Generate the new image
  prompt = "A realistic car running on the same lane, slightly changed background, cinematic high-quality lighting"
  negative_prompt = "low quality, distorted, cartoonish, unrealistic"
  output = pipe(
      prompt=prompt,
      image=input_image,
      strength=0.3,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]

  file_name = "generated_image_MR3_"+str(random.randint(10000, 99999))+".png"
  output.save(file_name)
  print("Generated image saved as"+file_name)

#Placing obstacles such as pedestrians, vehicles, or debris in the driving lane
def MR4(image_path):
  # Load your input image
  input_image_path = image_path  # Replace with your image path
  input_image = Image.open(input_image_path).convert("RGB")
  input_image = input_image.resize((1024, 512))  # High resolution for better quality

  # Generate the new image
  prompt = "A realistic car running on the same lane, slightly changed background, cinematic high-quality lighting"
  negative_prompt = "low quality, distorted, cartoonish, unrealistic"
  output = pipe(
      prompt=prompt,
      image=input_image,
      strength=0.3,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]

  file_name = "generated_image_MR4_"+str(random.randint(10000, 99999))+".png"
  output.save(file_name)
  print("Generated image saved as"+file_name)

#Increasing or decreasing the number and width of lanes.
def MR5(image_path):
  # Load your input image
  input_image_path = image_path  # Replace with your image path
  input_image = Image.open(input_image_path).convert("RGB")
  input_image = input_image.resize((1024, 512))  # High resolution for better quality

  # Generate the new image
  prompt = "A realistic car running on the same lane, slightly changed background, cinematic high-quality lighting"
  negative_prompt = "low quality, distorted, cartoonish, unrealistic"
  output = pipe(
      prompt=prompt,
      image=input_image,
      strength=0.3,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]

  file_name = "generated_image_MR5_"+str(random.randint(10000, 99999))+".png"
  output.save(file_name)
  print("Generated image saved as"+file_name)
