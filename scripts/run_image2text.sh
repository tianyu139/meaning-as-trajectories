# Prompt-free Results
python image2text_main.py llava trajectory --n_traj 20 --max_traj_len 20

# Prompted Results
python image2text_main.py llava trajectory --n_traj 20 --max_traj_len 20 --image_prompt "Describe this image." --text_prompt " This is a caption for an image. Describe this image." --post_image_prompt "The image shows" --post_text_prompt "The image shows"
