import functions_framework
import vertexai
import os
import time
import json
import random
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import storage
from enum import Enum
from werkzeug.datastructures import MultiDict

GCLOUD_PROJECT = os.environ.get('GCLOUD_PROJECT')
GCLOUD_LOCATION = os.environ.get('GCLOUD_LOCATION')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MODEL_NAME = os.environ.get('MODEL_NAME')
VALID_PASSWORD = os.environ.get('PASSWORD')

@functions_framework.http
def my_http(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # Phase 1: input validation
    errors = []
    if request.method != 'POST':
        errors.append("Only POST requests are accepted")

    form_data = MultiDict(request.form)

    password = form_data.get('password')
    if not password or password != VALID_PASSWORD :
        errors.append("Password is invalid")

    prompt = form_data.get('prompt', '')
    if not prompt or len(prompt) > 400:
        errors.append("Prompt is required and must not exceed 400 characters")

    negative_prompt = form_data.get('negative_prompt', '')
    if negative_prompt and len(negative_prompt) > 400:
        errors.append("Negative prompt must not exceed 400 characters")

    try:
        number_of_images = int(form_data.get('number_of_images', 1))
        if not 1 <= number_of_images <= 8:
            errors.append("Number of images must be between 1 and 8")
    except ValueError:
        errors.append("Invalid number of images")

    try:
        aspect_ratio = AspectRatio(form_data.get('aspect_ratio', ''))
    except ValueError:
        errors.append("Invalid aspect ratio")

    try:
        guidance_scale = float(form_data.get('guidance_scale', 11))
        if not 0 <= guidance_scale <= 21:
            errors.append("Guidance scale must be between 0 and 21")
    except ValueError:
        errors.append("Invalid guidance scale")

    seed = random.randint(0, 1_000_000)
    if 'seed' in form_data:
        try:
            seed = int(form_data['seed'])
        except ValueError:
            seed = random.randint(0, 1_000_000)

    try:
        safety_filter_level = SafetyFilterLevel(form_data.get('safety_filter_level', ''))
    except ValueError:
        errors.append("Invalid safety filter level")

    try:
        person_generation = PersonGeneration(form_data.get('person_generation', ''))
    except ValueError:
        errors.append("Invalid person generation option")

    if errors:
        return (json.dumps({"errors": errors}), 400, {'Content-Type': 'application/json'})

    print(f"Generating image(s) with prompt: {prompt}")

    # Phase 2: image generation
    vertexai.init(project=GCLOUD_PROJECT, location=GCLOUD_LOCATION)
    model = ImageGenerationModel.from_pretrained(MODEL_NAME)
    images = model.generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        number_of_images=number_of_images,
        language="en",
        seed=seed,
        add_watermark=False,
        aspect_ratio=aspect_ratio.value,
        guidance_scale=guidance_scale,
        safety_filter_level=safety_filter_level.value,
        person_generation=person_generation.value,
    )
    print(f"Generated {len(images.images)} image(s)")

    # Phase 3: uploading to cloud storage
    image_urls = []
    for image in images:
        image_path = str(time.time_ns() // 1_000_000) + ".png"
        image_urls.append("https://storage.googleapis.com/" + BUCKET_NAME + "/" + image_path)
        print(f"Created an image with size {len(image._image_bytes)} bytes in {image_path}")
        upload_blob_from_memory(image._loaded_bytes, image_path)

    headers = {"Access-Control-Allow-Origin": "*", 'Content-Type': 'application/json'}
    output = json.dumps({"image_urls": image_urls})
    print(f"Output: {output}")

    return output, 200, headers

def upload_blob_from_memory(contents, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents)


class AspectRatio(Enum):
    SQUARE = "1:1"
    PORTRAIT = "9:16"
    LANDSCAPE = "16:9"
    FOUR_THREE = "4:3"
    THREE_FOUR = "3:4"

class SafetyFilterLevel(Enum):
    BLOCK_MOST = "block_most"
    BLOCK_SOME = "block_some"
    BLOCK_FEW = "block_few"
    BLOCK_FEWEST = "block_fewest"

class PersonGeneration(Enum):
    DONT_ALLOW = "dont_allow"
    ALLOW_ALL = "allow_all"
    ALLOW_ADULT = "allow_adult"
