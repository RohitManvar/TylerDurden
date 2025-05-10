from flask import Flask, render_template, request, jsonify
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import random
import warnings
import json
from datetime import datetime
import re
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load intents from a separate file
def load_intents():
    try:
        with open('intents.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.info("intents.json not found, using default intents")
        return default_intents

# Default intents dictionary (same as before but with some improvements)
default_intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", 
                     "good evening", "hi there", "hey there", "what's up", "hello there", "sup", "yo"],
        "responses": ["Hello! How can I assist you today?", "Hi there! What can I help you with?", 
                     "Hey! What can I do for you?", "Greetings! How may I help you?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye", "farewell", "take care", "until next time", 
                    "bye bye", "catch you later", "have a good one", "so long", "gotta go", "leaving now"],
        "responses": ["Goodbye! Come back anytime.", "See you later! Have a great day!", 
                     "Farewell! Take care.", "Have a wonderful day ahead!"]
    },
    "gratitude": {
        "patterns": ["thank you", "thanks", "appreciate it", "thank you so much", "thanks a lot", 
                    "much appreciated", "thx", "thank u"],
        "responses": ["You're welcome!", "Happy to help!", "Glad I could assist.", 
                     "Anytime! Is there anything else you need?"]
    },
    "apologies": {
        "patterns": ["sorry", "my apologies", "apologize", "I'm sorry", "my bad", "excuse me"],
        "responses": ["No problem at all.", "It's alright.", "No need to apologize.", 
                     "That's okay. How can I help you now?"]
    },
    "positive_feedback": {
        "patterns": ["great job", "well done", "awesome", "fantastic", "amazing work", "excellent", 
                    "good bot", "love this", "you're smart"],
        "responses": ["Thank you! I appreciate your feedback.", "Glad to hear that!", 
                     "Thank you for the compliment!", "I'm happy I could meet your expectations."]
    },
    "negative_feedback": {
        "patterns": ["not good", "disappointed", "unsatisfied", "poor service", "needs improvement", 
                    "could be better", "you're wrong", "that's incorrect", "bad answer"],
        "responses": ["I'm sorry to hear that. How can I improve?", 
                     "I apologize for the inconvenience. Let me try to help better.", 
                     "Your feedback helps me improve. Could you tell me more about the issue?"]
    },
    "weather": {
        "patterns": ["what's the weather like", "weather forecast", "is it going to rain today", 
                    "temperature today", "weather report", "how's the weather", "is it sunny"],
        "responses": ["I don't have real-time weather data, but I can help you find a weather service to check.", 
                     "To get accurate weather information, I recommend checking a weather website or app.", 
                     "While I can't access current weather data, I'd be happy to help with other questions."]
    },
    "help": {
        "patterns": ["help", "can you help me", "I need assistance", "support", "what can you do", 
                    "how do you work", "your capabilities", "help me with"],
        "responses": ["I can help with various topics including greetings, jokes, recommendations, and more. What do you need assistance with?", 
                     "I'm here to assist you with information and answers. Just tell me what you're looking for.", 
                     "I can answer questions, provide information, or just chat. What would you like help with?"]
    },
    "time": {
        "patterns": ["what's the time", "current time", "time please", "what time is it", "tell me the time", 
                    "got the time", "check time"],
        "responses": ["It's [current_time].", "The current time is [current_time].", 
                     "Right now, it's [current_time]."]
    },
    "jokes": {
        "patterns": ["tell me a joke", "joke please", "got any jokes", "make me laugh", "be funny", 
                    "say something funny", "humor me"],
        "responses": ["Why don't we ever tell secrets on a farm? Because the potatoes have eyes and the corn has ears!", 
                     "What do you get when you cross a snowman and a vampire? Frostbite!", 
                     "Why was the math book sad? Because it had too many problems!", 
                     "What did the ocean say to the beach? Nothing, it just waved!", 
                     "Why don't scientists trust atoms? Because they make up everything!"]
    },
    "unknown": {
        "patterns": [],
        "responses": ["I'm not sure I understand. Could you rephrase that?", 
                     "I don't have information about that yet.", 
                     "I'm still learning. Could you try asking something else?", 
                     "I'm not able to help with that specific query. Is there something else I can assist with?"]
    }
}

# Global variables for the ML model
vectorizer = None
classifier = None
pipeline = None

def preprocess_text(text):
    """Clean and normalize text input"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except question marks
    text = re.sub(r'[^\w\s\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_model():
    """Train the intent classification model with optimized parameters"""
    global vectorizer, classifier, pipeline
    
    logger.info("Training the intent classification model...")
    
    # Load intents
    intents_data = load_intents()
    
    # Prepare training data
    training_data = []
    labels = []
    
    for intent, data in intents_data.items():
        if intent != "unknown":  # Skip the unknown intent for training
            for pattern in data['patterns']:
                training_data.append(preprocess_text(pattern))
                labels.append(intent)
    
    # Create pipeline with TF-IDF vectorizer and LinearSVC classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            tokenizer=nltk.word_tokenize,
            stop_words="english",
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )),
        ('classifier', LinearSVC(C=1.0, class_weight='balanced', max_iter=1000))
    ])
    
    # Split data for training and testing
    if len(training_data) > 10:  # Make sure we have enough data
        X_train, X_test, y_train, y_test = train_test_split(
            training_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = pipeline.predict(X_test)
        logger.info("Model training completed.")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    else:
        # If not enough data, train on all available data
        pipeline.fit(training_data, labels)
        logger.info("Model trained on limited data.")
    
    return pipeline

def predict_intent(user_input, confidence_threshold=0.4):
    """Predict the intent of user input with confidence threshold"""
    if not user_input:
        return "unknown", 0.0
    
    processed_input = preprocess_text(user_input)
    
    # Use the trained pipeline to predict
    try:
        # Get the predicted class
        intent = pipeline.predict([processed_input])[0]
        
        # Get the confidence score for LinearSVC
        # This is a bit tricky since LinearSVC doesn't have predict_proba
        decision_values = pipeline.named_steps['classifier'].decision_function([
            pipeline.named_steps['vectorizer'].transform([processed_input]).toarray()[0]
        ])
        
        # Find the max confidence score
        max_conf = max(decision_values[0])
        normalized_conf = (max_conf + 1) / 2  # Normalize to 0-1 range
        
        # Return unknown if confidence is below threshold
        if normalized_conf < confidence_threshold:
            return "unknown", normalized_conf
        
        return intent, normalized_conf
    
    except Exception as e:
        logger.error(f"Error predicting intent: {e}")
        return "unknown", 0.0

def get_response(user_input):
    """Process user message and get appropriate response"""
    # Check for empty input
    if not user_input or user_input.strip() == "":
        return "I didn't catch that. Could you please say something?"
    
    # Exit command
    if user_input.lower() in ["exit", "quit", "bye"]:
        return random.choice(default_intents["goodbye"]["responses"])
    
    # Special case for time intent
    if re.search(r'\btime\b', user_input.lower()):
        time_now = datetime.now().strftime("%H:%M")
        return f"The current time is {time_now}."
    
    # Predict intent
    intent, confidence = predict_intent(user_input)
    
    # Log the prediction
    logger.debug(f"User input: '{user_input}', Predicted intent: '{intent}', Confidence: {confidence:.2f}")
    
    # Get the response
    intents_data = load_intents()
    if intent in intents_data:
        responses = intents_data[intent]['responses']
        response = random.choice(responses)
        
        # Handle special placeholders in responses
        if "[current_time]" in response:
            current_time = datetime.now().strftime("%H:%M")
            response = response.replace("[current_time]", current_time)
        
        return response
    else:
        return random.choice(intents_data["unknown"]["responses"])

# Initialize the model when the server starts
@app.before_first_request
def initialize():
    global pipeline
    pipeline = train_model()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_input = request.json.get('message', '')
        response = get_response(user_input)
        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({'response': 'Sorry, I encountered an error processing your request.', 
                       'status': 'error', 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_endpoint():
    """Endpoint to manually retrain the model"""
    try:
        train_model()
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/healthcheck', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Service is running'})

if __name__ == '__main__':
    # Initialize the model before starting the server
    pipeline = train_model()
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)