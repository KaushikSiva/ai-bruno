# File: captioner.py

from transformers import pipeline
import os

# Initialize the pipeline once when the module is first imported.
# This makes the function more efficient as the model isn't loaded every time.
# The pipeline will handle caching the model.
try:
    image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    image_to_text_pipeline = None

def get_caption(image_path: str) -> str:
    """
    Generates a caption for a given image file path or URL.

    Args:
        image_path (str): The local file path or URL of the image.

    Returns:
        str: The generated caption or an error message.
    """
    if image_to_text_pipeline is None:
        return "Captioning service is unavailable."
    
    # Check if the path is a local file and if it exists
    if not image_path.startswith(('http://', 'https://')) and not os.path.exists(image_path):
        return f"Error: The image file '{image_path}' was not found."

    try:
        # Use the pipeline to generate a caption
        result = image_to_text_pipeline(image_path)
        return result[0]['generated_text']
    except Exception as e:
        return f"An error occurred during captioning: {e}"

# Example of how to use the function within this file for testing
if __name__ == "__main__":
    # Example with a URL
    url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    print(f"Captioning URL: {url}")
    print(f"Generated Caption: {get_caption(url)}\n")
    
    # Example with a local file (replace 'brazil.jpg' with your image)
    local_file = r"C:\Users\navee\Downloads\R9.jpg"
    print(f"Captioning local file: {local_file}")
    print(f"Generated Caption: {get_caption(local_file)}")