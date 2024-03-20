import requests
from diffusers import DiffusionPipeline
from transformers import BertForSequenceClassification, AutoTokenizer
import base64
from PIL import Image
import io
from flask import Flask, request, render_template


app = Flask(__name__,static_url_path='/static')
app.debug = True


class APIModel:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

    def query(self, payload):
        response = requests.post(self.url, headers=self.headers, json=payload)
        return response

class ImageAPI(APIModel):
    def __init__(self, url, headers):
        super().__init__(url, headers)

    def get_image(self, payload):
        response = requests.post(self.url, headers=self.headers, json=payload)
        img_data = response.content
        # Enregistrer l'image dans un fichier
        with open("static/output.png", "wb") as f:
            f.write(img_data)
        return img_data

class TextAPI(APIModel):
    def __init__(self, url, headers):
        super().__init__(url, headers)

    def get_text(self, payload):
        return self.query(payload).json()

class AudioAPI(APIModel):
    def __init__(self, url, headers):
        super().__init__(url, headers)

    def get_audio(self, payload):
        return self.query(payload).content

class CoM:
    def __init__(self):
        self.api_models = []
        self.model_id = "Nielzac/CoM_Small_AIL"
        self.model = BertForSequenceClassification.from_pretrained(self.model_id, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def add_api_model(self, api_model):
        self.api_models.append(api_model)

    def chatbot_response(self, user_input):
        # Tokenize the user input
        inputs = self.tokenizer(user_input, return_tensors="pt")

        # Make a prediction using the model
        outputs = self.model(**inputs)

        # Get the predicted class index
        predicted_class = outputs.logits.argmax().item()

        # Choose the API model based on the predicted class
        api_model = self.api_models[predicted_class]

        # Generate a response using the chosen API model
        if isinstance(api_model, TextAPI):
            response = api_model.get_text({"inputs": user_input})
            chosen_model = "Text model"
        elif isinstance(api_model, AudioAPI):
            response = api_model.get_audio({"inputs": user_input})
            chosen_model = "Audio model"
        elif isinstance(api_model, ImageAPI):
            response = api_model.get_image({"inputs": user_input})
            chosen_model = "Image model"
            response = user_input
        else:
            response = "Unknown API model type"
            chosen_model = "Unknown model"

        return response, chosen_model

com = CoM()

# Add the API models to the CoM instance
com.add_api_model(TextAPI("https://api-inference.huggingface.co/models/microsoft/phi-2", {"Authorization": "Bearer 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"}))
com.add_api_model(AudioAPI("https://api-inference.huggingface.co/models/procit001/nepali_male_v1", {"Authorization": "Bearer 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"}))
com.add_api_model(ImageAPI("https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5", {"Authorization": "Bearer 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"}))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send", methods=["POST"])
def send():
    user_input = request.form["user_input"]
    response, chosen_model = com.chatbot_response(user_input)
    return render_template("index.html", response=response, chosen_model=chosen_model)


if __name__ == "__main__":
    app.run()
