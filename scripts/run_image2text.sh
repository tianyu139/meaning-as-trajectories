# Prompt-free Results
python image2text_main.py llava trajectory

# Prompted Results
python image2text_main.py llava trajectory --image_prompt "Describe this image." --text_prompt " This is a caption for an image. Describe this image." --post_image_prompt "The image shows" --post_text_prompt "The image shows"
