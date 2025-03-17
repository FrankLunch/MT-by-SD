from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
from huggingface_hub import login
from config import Config
import os
# Log in with your Hugging Face token
login("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")  # Replace with your actual token

# Load the pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl",
    use_auth_token=True,  # Use token for authentication
    torch_dtype=torch.float16
)

# Check if CUDA is available and use appropriate device
# Get list of available GPUs with enough memory
available_gpus = []
for i in range(torch.cuda.device_count()):
    try:
        # Check if GPU has enough memory (e.g. 8GB)
        if torch.cuda.get_device_properties(i).total_memory > 8e9:  
            available_gpus.append(str(i))
    except:
        continue
        
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
pipe = pipe.to(device)
# if device == "cpu":
#     print("Warning: Running on CPU. This will be significantly slower than GPU.")


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
      strength=0.2,  # Lower strength to preserve original details(0.5)
      guidance_scale=10,  # Higher value for stronger alignment to prompt(8.5
      negative_prompt=negative_prompt,
  ).images[0]
  #output.save(new_image_name)
  return output

import random
import string
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

  import random
import string
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
import random
import string
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
import random
import string
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


import os
import pandas as pd
from shutil import copyfile


# List of folders to process
cfg = Config()
cfg.from_pyfile("Config_MTSD_My.py")
full_paths = [os.path.join(cfg.BASIC_DATA_PATH, folder) for folder in cfg.FOLDER_LIST]
print('processing paths is: ')
# Print the full paths
for path in full_paths:
    print(path)
folder_list = full_paths
#get the relationship list
mr_list = cfg.MR_LIST
#preprocess the data
# prompt: read the file /content/driving_log.csv, add a new column 'extract_num', for the lines that 'Crashed'==1 , select the countinuous lines , and also former 50 lines before this interval , all these are an interval give all these lines the value of 1, then for the next discontinuous interval(contains many lines) value is 2 , then is 3... others are 0 , save the csv

import pandas as pd

# Read the CSV file
for path in full_paths:
    df = pd.read_csv(path+'/driving_log.csv')
    # Add the 'extract_num' column initialized to 0
    df['extract_num'] = 0
    # Find indices where 'Crashed' is 1
    crashed_indices = df.index[df['Crashed'] == 1].tolist()
    interval_num = 0
    for i in range(len(crashed_indices)):
        current_index = crashed_indices[i]
        # Handle the case where the current index is within 50 lines of a previous interval
        if i > 0 and current_index - crashed_indices[i-1] <=50:
            start_index = max(0, crashed_indices[i-1] - 50)
            end_index = min(len(df), current_index + 1)

            df.loc[start_index:end_index, 'extract_num'] = interval_num
        else:
            interval_num += 1
            start_index = max(0, current_index - 50)
            end_index = min(len(df), current_index + 1)

            df.loc[start_index:end_index, 'extract_num'] = interval_num

    # Save the updated DataFrame to a new CSV file
    df.to_csv(os.path.join(path, 'driving_log_updated.csv'), index=False)

# Process each folder with generator
for mr in mr_list:
  for folder in folder_list:
      original_folder = folder
      new_folder = folder + '_' + mr
      # Create the new folder if it doesn't exist
      os.makedirs(new_folder, exist_ok=True)

      # Path to the driving log
      driving_log_path = os.path.join(original_folder, 'driving_log_updated.csv')
      if os.path.exists(driving_log_path):
          # Load CSV file
          df = pd.read_csv(driving_log_path)
          # Filter rows where 'Crashed' == 1
          crashed_df = df[df['extract_num'] != 0]
          #count
          num_rows = crashed_df.shape[0]
          count=0
          for _, row in crashed_df.iterrows():
              #to make name right
              original_image_path = row['center']
              original_image_path = str(original_image_path).replace('\\','/')[1:]
              original_image_full_path = os.path.join(original_folder, original_image_path)
              count = count+1
              if os.path.exists(original_image_full_path):
                  # Generate new image
                  print('run')
                  #new_image_output = MR1(original_image_full_path)
                  new_image_output = globals()[mr](original_image_full_path)
                  # Extract old image name and append "_MR1"
                  old_image_name = os.path.basename(original_image_full_path)
                  new_image_name = old_image_name.replace('.jpg', '_MR1.jpg')

                  # Define new image save path
                  new_image_full_path = os.path.join(new_folder, new_image_name)

                  # Save the new image (copying for now, replace with real save logic)
                  new_image_output.save(new_image_full_path)
                  print(f"Generated: {new_image_full_path}, {count}/{num_rows}")
      else:
          print(f"Skipping {folder}: driving_log.csv not found")

# # Load your input image
# input_image_path = "/content/1.jpg"  # Replace with your image path
# input_image = Image.open(input_image_path).convert("RGB")
# input_image = input_image.resize((512, 512))

# # Generate the new image
# prompt = "keep the lane same , change the tree color to red"
# output = pipe(
#     prompt=prompt,
#     image=input_image,
#     strength=0.7,
#     guidance_scale=7.5,
#     negative_prompt="low quality, blurry"
# ).images[0]

# # Save the generated image
# output.save("generated_image.png")
# print("Generated image saved as 'generated_image.png'")



