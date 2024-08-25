# import streamlit as st
# import cv2
# import numpy as np
# import openai
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from msrest.authentication import CognitiveServicesCredentials
# from PIL import Image, ImageEnhance
# import io
# import time
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Set your Azure Computer Vision credentials
# subscription_key = "" #os.getenv("VISION_KEY")
# endpoint = ""   #os.getenv("VISION_ENDPOINT")

# # Check if environment variables are set
# if not subscription_key:
#     st.error("Environment variable 'VISION_KEY' is not set.")
# if not endpoint:
#     st.error("Environment variable 'VISION_ENDPOINT' is not set.")

# # Initialize Computer Vision client
# computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# # Set your OpenAI API key
# openai.api_key = ''   # Replace with your actual OpenAI API key

# # Function to enhance image
# def enhance_image(image):
#     try:
#         image = image.convert("RGB")
#         enhancer = ImageEnhance.Contrast(image)
#         image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
#         logging.info("Image enhancement completed.")
#         return image_enhanced
#     except Exception as e:
#         logging.error(f"Error enhancing image: {e}")
#         raise

# # Function to analyze image with Azure Computer Vision
# def analyze_image(image):
#     try:
#         image = enhance_image(image)
#         image_stream = io.BytesIO()
#         image.save(image_stream, format='JPEG')
#         image_stream.seek(0)
#         read_response = computervision_client.read_in_stream(image_stream, raw=True)
#         read_operation_location = read_response.headers["Operation-Location"]
#         operation_id = read_operation_location.split("/")[-1]
#         while True:
#             read_result = computervision_client.get_read_result(operation_id)
#             if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
#                 break
#             time.sleep(1)
#         text_results = []
#         if read_result.status == OperationStatusCodes.succeeded:
#             for page in read_result.analyze_result.read_results:
#                 for line in page.lines:
#                     bounding_box = {
#                         "left": int(min(line.bounding_box[0::2])),
#                         "top": int(min(line.bounding_box[1::2])),
#                         "width": int(max(line.bounding_box[0::2]) - min(line.bounding_box[0::2])),
#                         "height": int(max(line.bounding_box[1::2]) - min(line.bounding_box[1::2]))
#                     }
#                     text_results.append({
#                         "text": line.text,
#                         "bounding_box": bounding_box
#                     })
#         logging.info(f"Text analysis results: {text_results}")
#         return text_results
#     except Exception as e:
#         logging.error(f"Error analyzing image: {e}")
#         raise

# # Streamlit application
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read the file content
#     file_bytes = uploaded_file.read()

#     # Load image with PIL
#     image = Image.open(io.BytesIO(file_bytes))

#     # Load image with OpenCV
#     image_cv2 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if image_cv2 is None:
#         st.error("Error loading image. Please try again with a different file.")
#     else:
#         # Analyze image with Azure Computer Vision
#         text_results = analyze_image(image)

#         # Draw bounding boxes on the original image without text labels
#         for result in text_results:
#             bbox = result["bounding_box"]
#             top_left = (int(bbox["left"]), int(bbox["top"]))
#             bottom_right = (int(bbox["left"] + bbox["width"]), int(bbox["top"] + bbox["height"]))
#             cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)

#         # Convert image from BGR to RGB format
#         image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

#         # Display the output image with bounding boxes
#         st.image(image_cv2_rgb, caption='Processed Image with Detected Text', use_column_width=True)

#         # Display the extracted text
#         detected_text = " ".join([result["text"] for result in text_results])
#         # st.write("### Extracted Text")
#         # st.write(detected_text)

#         # Use LLM to classify the extracted text
#         structured_text = f"""
#         Extract and classify the following text into the required details:

#         Text:
#         {detected_text}

#         Provide the output in the following format:
#         Known context = [context]
#         Brand/Mark = [brand]
#         Keywords = [keywords]
#         Detected Objects = [objects]
#         """
#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo-instruct",
#             prompt=structured_text,
#             max_tokens=200,
#             n=1,
#             stop=None,
#             temperature=0.3,
#         )

#         # Extract the classification results
#         result = response.choices[0].text.strip()

#         # Ensure each classification result is on a separate line with spaces in between
#         known_context = ""
#         brand_mark = ""
#         keywords = ""
#         detected_objects = ""

#         for line in result.split("\n"):
#             if "Known context =" in line:
#                 known_context = line.split("=", 1)[1].strip()
#             elif "Brand/Mark =" in line:
#                 brand_mark = line.split("=", 1)[1].strip()
#             elif "Keywords =" in line:
#                 keywords = line.split("=", 1)[1].strip()
#             elif "Detected Objects =" in line:
#                 detected_objects = line.split("=", 1)[1].strip()

#         # Display the classification results in a formatted layout
#         st.write("### Outcome")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**1** Visual Search")
#         with col2:
#             st.markdown(f"**Known context =** {known_context}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**2** Logo / Mark Detection")
#         with col2:
#             st.markdown(f"**Brand/Mark =** {brand_mark}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**3** Text Detection")
#         with col2:
#             st.markdown(f"**Keywords =** {keywords}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**4** Custom Object Detection")
#         with col2:
#             st.markdown(f"**Detected Objects =** {detected_objects}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import streamlit as st
# import cv2
# import numpy as np
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from azure.ai.textanalytics import TextAnalyticsClient
# from azure.core.credentials import AzureKeyCredential
# from msrest.authentication import CognitiveServicesCredentials
# from PIL import Image, ImageEnhance
# import io
# import time
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Set your Azure Computer Vision credentials
# subscription_key = ""
# endpoint = ""

# # Set your Azure Text Analytics credentials
# text_analytics_key = ""
# text_analytics_endpoint = """
# # Check if environment variables are set
# if not subscription_key:
#     st.error("Environment variable 'VISION_KEY' is not set.")
# if not endpoint:
#     st.error("Environment variable 'VISION_ENDPOINT' is not set.")
# if not text_analytics_key:
#     st.error("Environment variable 'TEXT_ANALYTICS_KEY' is not set.")
# if not text_analytics_endpoint:
#     st.error("Environment variable 'TEXT_ANALYTICS_ENDPOINT' is not set.")

# # Initialize Computer Vision client
# computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# # Initialize Text Analytics client
# text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_key))

# # Function to enhance image
# def enhance_image(image):
#     try:
#         image = image.convert("RGB")
#         enhancer = ImageEnhance.Contrast(image)
#         image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
#         logging.info("Image enhancement completed.")
#         return image_enhanced
#     except Exception as e:
#         logging.error(f"Error enhancing image: {e}")
#         raise

# # Function to analyze image with Azure Computer Vision
# def analyze_image(image):
#     try:
#         image = enhance_image(image)
#         image_stream = io.BytesIO()
#         image.save(image_stream, format='JPEG')
#         image_stream.seek(0)
#         read_response = computervision_client.read_in_stream(image_stream, raw=True)
#         read_operation_location = read_response.headers["Operation-Location"]
#         operation_id = read_operation_location.split("/")[-1]
#         while True:
#             read_result = computervision_client.get_read_result(operation_id)
#             if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
#                 break
#             time.sleep(1)
#         text_results = []
#         if read_result.status == OperationStatusCodes.succeeded:
#             for page in read_result.analyze_result.read_results:
#                 for line in page.lines:
#                     bounding_box = {
#                         "left": int(min(line.bounding_box[0::2])),
#                         "top": int(min(line.bounding_box[1::2])),
#                         "width": int(max(line.bounding_box[0::2]) - min(line.bounding_box[0::2])),
#                         "height": int(max(line.bounding_box[1::2]) - min(line.bounding_box[1::2]))
#                     }
#                     text_results.append({
#                         "text": line.text,
#                         "bounding_box": bounding_box
#                     })
#         logging.info(f"Text analysis results: {text_results}")
#         return text_results
#     except Exception as e:
#         logging.error(f"Error analyzing image: {e}")
#         raise
# # Function to classify text using Azure Text Analytics
# def classify_text(text):
#     try:
#         documents = [text]
#         response = text_analytics_client.analyze_sentiment(documents=documents)
#         results = {
#             "known_context": "",
#             "brand_mark": "",
#             "keywords": "",
#             "detected_objects": ""
#         }

#         for idx, document in enumerate(response):
#             if not document.is_error:
#                 results["known_context"] = document.sentiment

#                 # Analyze entities to get key phrases
#                 entities_result = text_analytics_client.recognize_entities(documents=documents)[0]
#                 results["keywords"] = ", ".join([entity.text for entity in entities_result.entities if entity.category == "KeyPhrase"])

#                 for entity in entities_result.entities:
#                     if entity.category == "Brand":
#                         results["brand_mark"] += entity.text + " "
#                     if entity.category == "Product":
#                         results["detected_objects"] += entity.text + " "

#         logging.info(f"Text classification results: {results}")
#         return results

#     except Exception as e:
#         logging.error(f"Error classifying text: {e}")
#         raise
# # Streamlit application
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read the file content
#     file_bytes = uploaded_file.read()

#     # Load image with PIL
#     image = Image.open(io.BytesIO(file_bytes))

#     # Load image with OpenCV
#     image_cv2 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if image_cv2 is None:
#         st.error("Error loading image. Please try again with a different file.")
#     else:
#         # Analyze image with Azure Computer Vision
#         text_results = analyze_image(image)

#         # Draw bounding boxes on the original image without text labels
#         for result in text_results:
#             bbox = result["bounding_box"]
#             top_left = (int(bbox["left"]), int(bbox["top"]))
#             bottom_right = (int(bbox["left"] + bbox["width"]), int(bbox["top"] + bbox["height"]))
#             cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)

#         # Convert image from BGR to RGB format
#         image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

#         # Display the output image with bounding boxes
#         st.image(image_cv2_rgb, caption='Processed Image with Detected Text', use_column_width=True)

#         # Display the extracted text
#         detected_text = " ".join([result["text"] for result in text_results])
#         st.write("### Extracted Text")
#         st.write(detected_text)

#         # Classify the extracted text using Azure Text Analytics
#         classification_results = classify_text(detected_text)

#         # Display the classification results in a formatted layout
#         st.write("### Outcome")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**1** Visual Search")
#         with col2:
#             st.markdown(f"**Known context =** {classification_results['known_context']}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**2** Logo / Mark Detection")
#         with col2:
#             st.markdown(f"**Brand/Mark =** {classification_results['brand_mark']}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**3** Text Detection")
#         with col2:
#             st.markdown(f"**Keywords =** {classification_results['keywords']}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**4** Custom Object Detection")
#         with col2:
#             st.markdown(f"**Detected Objects =** {classification_results['detected_objects']}")


# ---------------------------------------------------------------------------------------------


# import streamlit as st
# import cv2
# import numpy as np
# import requests
# import base64
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from msrest.authentication import CognitiveServicesCredentials
# from PIL import Image, ImageEnhance
# import io
# import time
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Set your Azure Computer Vision credentials
# subscription_key = os.getenv("VISION_KEY")
# endpoint = os.getenv("VISION_ENDPOINT")

# # Check if environment variables are set
# if not subscription_key:
#     st.error("Environment variable 'VISION_KEY' is not set.")
# if not endpoint:
#     st.error("Environment variable 'VISION_ENDPOINT' is not set.")

# # Initialize Computer Vision client
# computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# # Set your Azure AI Studio Chat API key and endpoint
# GPT4V_KEY = """
# # GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT")
# GPT4V_ENDPOINT = """
# if not GPT4V_KEY:
#     st.error("Environment variable 'GPT4V_KEY' is not set.")
# if not GPT4V_ENDPOINT:
#     st.error("Environment variable 'GPT4V_ENDPOINT' is not set.")

# # Function to enhance image
# def enhance_image(image):
#     try:
#         image = image.convert("RGB")
#         enhancer = ImageEnhance.Contrast(image)
#         image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
#         logging.info("Image enhancement completed.")
#         return image_enhanced
#     except Exception as e:
#         logging.error(f"Error enhancing image: {e}")
#         raise

# # Function to analyze image with Azure Computer Vision
# def analyze_image(image):
#     try:
#         image = enhance_image(image)
#         image_stream = io.BytesIO()
#         image.save(image_stream, format='JPEG')
#         image_stream.seek(0)
#         read_response = computervision_client.read_in_stream(image_stream, raw=True)
#         read_operation_location = read_response.headers["Operation-Location"]
#         operation_id = read_operation_location.split("/")[-1]
#         while True:
#             read_result = computervision_client.get_read_result(operation_id)
#             if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
#                 break
#             time.sleep(1)
#         text_results = []
#         if read_result.status == OperationStatusCodes.succeeded:
#             for page in read_result.analyze_result.read_results:
#                 for line in page.lines:
#                     bounding_box = {
#                         "left": int(min(line.bounding_box[0::2])),
#                         "top": int(min(line.bounding_box[1::2])),
#                         "width": int(max(line.bounding_box[0::2]) - min(line.bounding_box[0::2])),
#                         "height": int(max(line.bounding_box[1::2]) - min(line.bounding_box[1::2]))
#                     }
#                     text_results.append({
#                         "text": line.text,
#                         "bounding_box": bounding_box
#                     })
#         logging.info(f"Text analysis results: {text_results}")
#         return text_results
#     except Exception as e:
#         logging.error(f"Error analyzing image: {e}")
#         raise

# # Function to use Azure AI Studio Chat to classify text
# def classify_text_with_azure_ai_studio(text):
#     try:
#         payload = {
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": "You are an AI assistant that helps people find information."
#                         }
#                     ]
#                 },
#                 {
#                     "role": "user",
#                     "content": text
#                 }
#             ],
#             "temperature": 0.7,
#             "top_p": 0.95,
#             "max_tokens": 800
#         }
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": GPT4V_KEY,
#         }
#         response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()
#     except requests.RequestException as e:
#         logging.error(f"Error calling Azure AI Studio Chat API: {e}")
#         raise

# # Streamlit application
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read the file content
#     file_bytes = uploaded_file.read()

#     # Load image with PIL
#     image = Image.open(io.BytesIO(file_bytes))

#     # Load image with OpenCV
#     image_cv2 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if image_cv2 is None:
#         st.error("Error loading image. Please try again with a different file.")
#     else:
#         # Analyze image with Azure Computer Vision
#         text_results = analyze_image(image)

#         # Draw bounding boxes on the original image without text labels
#         for result in text_results:
#             bbox = result["bounding_box"]
#             top_left = (int(bbox["left"]), int(bbox["top"]))
#             bottom_right = (int(bbox["left"] + bbox["width"]), int(bbox["top"] + bbox["height"]))
#             cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)

#         # Convert image from BGR to RGB format
#         image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

#         # Display the output image with bounding boxes
#         st.image(image_cv2_rgb, caption='Processed Image with Detected Text', use_column_width=True)

#         # Display the extracted text
#         detected_text = " ".join([result["text"] for result in text_results])

#         # Use Azure AI Studio Chat to classify the extracted text
#         classification_result = classify_text_with_azure_ai_studio(detected_text)

#         # Extract the classification results
#         result = classification_result["choices"][0]["message"]["content"]

#         # Ensure each classification result is on a separate line with spaces in between
#         known_context = ""
#         brand_mark = ""
#         keywords = ""
#         detected_objects = ""

#         for line in result.split("\n"):
#             if "Known context =" in line:
#                 known_context = line.split("=", 1)[1].strip()
#             elif "Brand/Mark =" in line:
#                 brand_mark = line.split("=", 1)[1].strip()
#             elif "Keywords =" in line:
#                 keywords = line.split("=", 1)[1].strip()
#             elif "Detected Objects =" in line:
#                 detected_objects = line.split("=", 1)[1].strip()

#         # Display the classification results in a formatted layout
#         st.write("### Outcome")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**1** Visual Search")
#         with col2:
#             st.markdown(f"**Known context =** {known_context}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**2** Logo / Mark Detection")
#         with col2:
#             st.markdown(f"**Brand/Mark =** {brand_mark}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**3** Text Detection")
#         with col2:
#             st.markdown(f"**Keywords =** {keywords}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**4** Custom Object Detection")
#         with col2:
#             st.markdown(f"**Detected Objects =** {detected_objects}")

# __________________________________________________________________________________________________________
# import streamlit as st
# import cv2
# import numpy as np
# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from azure.core.credentials import AzureKeyCredential
# from msrest.authentication import CognitiveServicesCredentials
# from PIL import Image, ImageEnhance
# import io
# import time
# import logging
# import os
# import requests

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Set your Azure Computer Vision credentials
# subscription_key = ""
# endpoint = ""

# # Set your Gemini API key
# gemini_api_key = ""

# # Check if environment variables are set
# if not subscription_key:
#     st.error("Environment variable 'VISION_KEY' is not set.")
# if not endpoint:
#     st.error("Environment variable 'VISION_ENDPOINT' is not set.")
# if not gemini_api_key:
#     st.error("Environment variable 'GEMINI_API_KEY' is not set.")

# # Initialize Computer Vision client
# computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# # Function to enhance image
# def enhance_image(image):
#     try:
#         image = image.convert("RGB")
#         enhancer = ImageEnhance.Contrast(image)
#         image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
#         logging.info("Image enhancement completed.")
#         return image_enhanced
#     except Exception as e:
#         logging.error(f"Error enhancing image: {e}")
#         raise

# # Function to analyze image with Azure Computer Vision
# def analyze_image(image):
#     try:
#         image = enhance_image(image)
#         image_stream = io.BytesIO()
#         image.save(image_stream, format='JPEG')
#         image_stream.seek(0)
#         read_response = computervision_client.read_in_stream(image_stream, raw=True)
#         read_operation_location = read_response.headers["Operation-Location"]
#         operation_id = read_operation_location.split("/")[-1]
#         while True:
#             read_result = computervision_client.get_read_result(operation_id)
#             if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
#                 break
#             time.sleep(1)
#         text_results = []
#         if read_result.status == OperationStatusCodes.succeeded:
#             for page in read_result.analyze_result.read_results:
#                 for line in page.lines:
#                     bounding_box = {
#                         "left": int(min(line.bounding_box[0::2])),
#                         "top": int(min(line.bounding_box[1::2])),
#                         "width": int(max(line.bounding_box[0::2]) - min(line.bounding_box[0::2])),
#                         "height": int(max(line.bounding_box[1::2]) - min(line.bounding_box[1::2]))
#                     }
#                     text_results.append({
#                         "text": line.text,
#                         "bounding_box": bounding_box
#                     })
#         logging.info(f"Text analysis results: {text_results}")
#         return text_results
#     except Exception as e:
#         logging.error(f"Error analyzing image: {e}")
#         raise

# # Function to classify text into predefined categories
# def classify_text_gemini(text):
#     try:
#         categories = {
#             "Visual Search": ["image", "visual", "search"],
#             "Logo / Mark Detection": ["logo", "mark", "brand"],
#             "Text Detection": ["text", "OCR", "read"],
#             "Custom Object Detection": ["object", "detect", "custom"]
#         }

#         results = {
#             "known_context": "N/A",
#             "brand_mark": "N/A",
#             "keywords": "N/A",
#             "detected_objects": "N/A"
#         }

#         for category, keywords in categories.items():
#             for keyword in keywords:
#                 if keyword.lower() in text.lower():
#                     results[category.lower().replace(" ", "_")] = "Yes"

#         logging.info(f"Text classification results: {results}")
#         return results
#     except Exception as e:
#         logging.error(f"Error classifying text: {e}")
#         raise

# # Streamlit application
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read the file content
#     file_bytes = uploaded_file.read()

#     # Load image with PIL
#     image = Image.open(io.BytesIO(file_bytes))

#     # Load image with OpenCV
#     image_cv2 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

#     if image_cv2 is None:
#         st.error("Error loading image. Please try again with a different file.")
#     else:
#         # Analyze image with Azure Computer Vision
#         text_results = analyze_image(image)

#         # Draw bounding boxes on the original image without text labels
#         for result in text_results:
#             bbox = result["bounding_box"]
#             top_left = (int(bbox["left"]), int(bbox["top"]))
#             bottom_right = (int(bbox["left"] + bbox["width"]), int(bbox["top"] + bbox["height"]))
#             cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)

#         # Convert image from BGR to RGB format
#         image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

#         # Display the output image with bounding boxes
#         st.image(image_cv2_rgb, caption='Processed Image with Detected Text', use_column_width=True)

#         # Display the extracted text
#         detected_text = " ".join([result["text"] for result in text_results])
#         st.write("### Extracted Text")
#         st.write(detected_text)

#         # Classify the extracted text using Gemini API
#         classification_results = classify_text_gemini(detected_text)

#         # Display the classification results in a formatted layout
#         st.write("### Outcome")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**1** Visual Search")
#         with col2:
#             st.markdown(f"**Known context =** {classification_results.get('known_context', 'N/A')}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**2** Logo / Mark Detection")
#         with col2:
#             st.markdown(f"**Brand/Mark =** {classification_results.get('brand_mark', 'N/A')}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**3** Text Detection")
#         with col2:
#             st.markdown(f"**Keywords =** {classification_results.get('keywords', 'N/A')}")

#         st.markdown("---")

#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.markdown("**4** Custom Object Detection")
#         with col2:
#             st.markdown(f"**Detected Objects =** {classification_results.get('detected_objects', 'N/A')}")


import streamlit as st
import cv2
import numpy as np
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.core.credentials import AzureKeyCredential
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageEnhance
import io
import time
import logging
import os
import requests
from groq import Groq
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set your Azure Computer Vision credentials
load_dotenv()  # Load environment variables from .env file

subscription_key = os.getenv('sub_key')
endpoint = os.getenv('end_point')
# gemini_api_key = os.getenv('gemini_api')
groq_api_key = os.getenv('groq_api')

# Check if environment variables are set
if not subscription_key:
    st.error("Environment variable 'VISION_KEY' is not set.")
if not endpoint:
    st.error("Environment variable 'VISION_ENDPOINT' is not set.")
# if not gemini_api_key:
#     st.error("Environment variable 'GEMINI_API_KEY' is not set.")
# if not groq_api_key:
#     st.error("Environment variable 'GROQ_API_KEY' is not set.")


# Add title and description
st.title("Object Detection and Text Classification")
st.write("""
Welcome to the Object Detection and Text Classification app.
Upload an image, and we will analyze it using Computer Vision and classify the extracted text
""")
# Initialize Computer Vision client
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

# Initialize GROQ client
client = Groq(api_key=groq_api_key)

# Function to enhance image


def enhance_image(image):
    try:
        image = image.convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        image_enhanced = enhancer.enhance(2.0)  # Enhance contrast
        logging.info("Image enhancement completed.")
        return image_enhanced
    except Exception as e:
        logging.error(f"Error enhancing image: {e}")
        raise

# Function to analyze image with Azure Computer Vision


def analyze_image(image):
    try:
        image = enhance_image(image)
        image_stream = io.BytesIO()
        image.save(image_stream, format='JPEG')
        image_stream.seek(0)
        read_response = computervision_client.read_in_stream(
            image_stream, raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)
        text_results = []
        if read_result.status == OperationStatusCodes.succeeded:
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    bounding_box = {
                        "left": int(min(line.bounding_box[0::2])),
                        "top": int(min(line.bounding_box[1::2])),
                        "width": int(max(line.bounding_box[0::2]) - min(line.bounding_box[0::2])),
                        "height": int(max(line.bounding_box[1::2]) - min(line.bounding_box[1::2]))
                    }
                    text_results.append({
                        "text": line.text,
                        "bounding_box": bounding_box
                    })
        logging.info(f"Text analysis results: {text_results}")
        return text_results
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        raise

# Function to classify text into predefined categories using GROQ API


def classify_text_groq(text_to_classify):
    try:
        # Format the prompt for GROQ
        prompt = f"""
        Extract and classify the following text into the required details:

        Text:
        {text_to_classify}

        Provide the output in the following format:

        Known context = [context]

        Brand/Mark = [brand]

        Keywords = [keywords]

        Detected Objects = [objects]
        """

        # Call GROQ API to classify the text
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )

        # Extract the result
        result = chat_completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        logging.error(f"Error classifying text with GROQ: {e}")
        raise


# Streamlit application
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file content
    file_bytes = uploaded_file.read()

    # Load image with PIL
    image = Image.open(io.BytesIO(file_bytes))

    # Load image with OpenCV
    image_cv2 = cv2.imdecode(np.frombuffer(
        file_bytes, np.uint8), cv2.IMREAD_COLOR)

    if image_cv2 is None:
        st.error("Error loading image. Please try again with a different file.")
    else:
        # Analyze image with Azure Computer Vision
        text_results = analyze_image(image)

        # Draw bounding boxes on the original image without text labels
        for result in text_results:
            bbox = result["bounding_box"]
            top_left = (int(bbox["left"]), int(bbox["top"]))
            bottom_right = (int(bbox["left"] + bbox["width"]),
                            int(bbox["top"] + bbox["height"]))
            cv2.rectangle(image_cv2, top_left, bottom_right, (0, 255, 0), 2)

        # Convert image from BGR to RGB format
        image_cv2_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Display the output image with bounding boxes
        st.image(image_cv2_rgb, caption='Processed Image with Detected Text',
                 use_column_width=True)

        # Display the extracted text
        detected_text = " ".join([result["text"] for result in text_results])
        # st.write("### Extracted Text")
        # st.write(detected_text)

        # Classify the extracted text using GROQ API
        classification_results = classify_text_groq(detected_text)

        # Display the classification results in a formatted layout
        st.write("### Outcome")
        st.write(classification_results)
