from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from sklearn.model_selection import train_test_split
import random
import warnings
import threading
import json
from datetime import datetime
import os

# Download NLTK data at startup
nltk.download('punkt', quiet=True)

# Suppress warnings
warnings.simplefilter('ignore')

app = Flask(__name__)

intents = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hey there", "what's up", "hello there"],
        "responses": ["Hello! How can I assist you?", "Hi there!", "Hey! What can I do for you?", "Howdy! What brings you here?", "Greetings! How may I help you?", "Good morning! How can I be of service?", "Good afternoon! What do you need assistance with?", "Good evening! How may I assist you?", "Hey there! How can I help?", "Hi! What's on your mind?", "Hello there! How can I assist you today?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye", "farewell", "take care", "until next time", "bye bye", "catch you later", "have a good one", "so long"],
        "responses": ["Goodbye!", "See you later!", "Have a great day!", "Farewell! Take care.", "Goodbye! Until next time.", "Take care! Have a wonderful day.", "Bye bye!", "Catch you later!", "Have a good one!", "So long!"]
    },
    "gratitude": {
        "patterns": ["thank you", "thanks", "appreciate it", "thank you so much", "thanks a lot", "much appreciated"],
        "responses": ["You're welcome!", "Happy to help!", "Glad I could assist.", "Anytime!", "You're welcome! Have a great day.", "No problem!"]
    },
    "apologies": {
        "patterns": ["sorry", "my apologies", "apologize", "I'm sorry"],
        "responses": ["No problem at all.", "It's alright.", "No need to apologize.", "That's okay.", "Don't worry about it.", "Apology accepted."]
    },
    "positive_feedback": {
        "patterns": ["great job", "well done", "awesome", "fantastic", "amazing work", "excellent"],
        "responses": ["Thank you! I appreciate your feedback.", "Glad to hear that!", "Thank you for the compliment!", "I'm glad I could meet your expectations.", "Your words motivate me!", "Thank you for your kind words."]
    },
    "negative_feedback": {
        "patterns": ["not good", "disappointed", "unsatisfied", "poor service", "needs improvement", "could be better"],
        "responses": ["I'm sorry to hear that. Can you please provide more details so I can assist you better?", "I apologize for the inconvenience. Let me help resolve the issue.", "I'm sorry you're not satisfied. Please let me know how I can improve.", "Your feedback is valuable. I'll work on improving."]
    },
    "weather": {
        "patterns": ["what's the weather like?", "weather forecast", "is it going to rain today?", "temperature today", "weather report"],
        "responses": ["The weather today is [weather_description].", "Currently, it's [temperature] degrees with [weather_description].", "The forecast predicts [weather_forecast].", "It might rain today. Don't forget your umbrella!", "The temperature today is [temperature] degrees."]
    },
    "help": {
        "patterns": ["help", "can you help me?", "I need assistance", "support"],
        "responses": ["Sure, I'll do my best to assist you.", "Of course, I'm here to help!", "How can I assist you?", "I'll help you with your query."]
    },
    "time": {
        "patterns": ["what's the time?", "current time", "time please", "what time is it?"],
        "responses": ["It's [current_time].", "The current time is [current_time].", "Right now, it's [current_time]."]
    },
    "jokes": {
        "patterns": ["tell me a joke", "joke please", "got any jokes?", "make me laugh"],
        "responses": ["Why don't we ever tell secrets on a farm? Because the potatoes have eyes and the corn has ears!", "What do you get when you cross a snowman and a vampire? Frostbite!", "Why was the math book sad? Because it had too many problems!"]
    },
    "music": {
        "patterns": ["play music", "music please", "song recommendation", "music suggestion"],
        "responses": ["Sure, playing some music for you!", "Here's a song you might like: [song_name]", "How about some music?"]
    },
    "food": {
        "patterns": ["recommend a restaurant", "food places nearby", "what's good to eat?", "restaurant suggestion"],
        "responses": ["Sure, here are some recommended restaurants: [restaurant_names]", "Hungry? Let me find some good food places for you!", "I can suggest some great places to eat nearby."]
    },
    "news": {
        "patterns": ["latest news", "news updates", "what's happening?", "current events"],
        "responses": ["Let me fetch the latest news for you.", "Here are the top headlines: [news_headlines]", "Stay updated with the latest news!"]
    },
    "movies": {
        "patterns": ["movie suggestions", "recommend a movie", "what should I watch?", "best movies"],
        "responses": ["How about watching [movie_name]?", "Here's a movie suggestion for you.", "Let me recommend some great movies!"]
    },
    "sports": {
        "patterns": ["sports news", "score updates", "latest sports events", "upcoming games"],
        "responses": ["I'll get you the latest sports updates.", "Stay updated with the current sports events!", "Let me check the sports scores for you."]
    },
    "gaming": {
        "patterns": ["video game recommendations", "best games to play", "recommend a game", "gaming suggestions"],
        "responses": ["How about trying out [game_name]?", "Here are some gaming suggestions for you!", "Let me recommend some fun games to play!"]
    },
    "tech_support": {
        "patterns": ["technical help", "computer issues", "troubleshooting", "IT support"],
        "responses": ["I can assist with technical issues. What problem are you facing?", "Let's troubleshoot your technical problem together.", "Tell me about the technical issue you're experiencing."]
    },
    "book_recommendation": {
        "patterns": ["recommend a book", "good books to read", "book suggestions", "what should I read?"],
        "responses": ["How about reading [book_title]?", "I've got some great book recommendations for you!", "Let me suggest some interesting books for you to read."]
    },
    "fitness_tips": {
        "patterns": ["fitness advice", "workout tips", "exercise suggestions", "healthy habits"],
        "responses": ["Staying fit is important! Here are some fitness tips: [fitness_tips]", "I can help you with workout suggestions and fitness advice.", "Let me provide some exercise recommendations for you."]
    },
    "travel_recommendation": {
        "patterns": ["travel suggestions", "places to visit", "recommend a destination", "travel ideas"],
        "responses": ["Looking for travel recommendations? Here are some great destinations: [travel_destinations]", "I can suggest some amazing places for your next travel adventure!", "Let me help you with travel destination ideas."]
    },
    "education": {
        "patterns": ["learning resources", "study tips", "education advice", "academic help"],
        "responses": ["I can assist with educational queries. What subject are you studying?", "Let's explore learning resources together.", "Tell me about your educational goals or questions."]
    },
    "pet_advice": {
        "patterns": ["pet care tips", "animal advice", "pet health", "taking care of pets"],
        "responses": ["Pets are wonderful! Here are some pet care tips: [pet_care_tips]", "I can provide advice on pet health and care.", "Let's talk about your pet and their well-being."]
    },
    "shopping": {
        "patterns": ["online shopping", "buying something", "shopping advice", "product recommendations"],
        "responses": ["I can help you with online shopping. What are you looking to buy?", "Let's find the perfect item for you!", "Tell me what you're interested in purchasing."]
    },
    "career_advice": {
        "patterns": ["job search help", "career guidance", "career change advice", "professional development"],
        "responses": ["I can provide career advice. What specific guidance do you need?", "Let's explore career opportunities together.", "Tell me about your career goals or concerns."]
    },
    "relationship_advice": {
        "patterns": ["relationship help", "love advice", "dating tips", "relationship problems"],
        "responses": ["Relationships can be complex. How can I assist you?", "I can offer advice on relationships and dating.", "Tell me about your relationship situation."]
    },
    "mental_health": {
        "patterns": ["mental health support", "coping strategies", "stress relief tips", "emotional well-being"],
        "responses": ["Mental health is important. How can I support you?", "I can provide guidance for managing stress and emotions.", "Let's talk about strategies for maintaining mental well-being."]
    },
    "language_learning": {
        "patterns": ["language learning tips", "language practice", "learning new languages", "language study advice"],
        "responses": ["Learning a new language can be exciting! How can I assist you?", "I can help with language learning tips and practice.", "Tell me which language you're interested in learning."]
    },
    "finance_advice": {
        "patterns": ["financial planning help", "money management tips", "investment advice", "budgeting assistance"],
        "responses": ["I can provide guidance on financial matters. What specific advice do you need?", "Let's discuss your financial goals and plans.", "Tell me about your financial situation or goals."]
    },
}

# Training the model
training_data = []
labels = []

for intent, data in intents.items():
    for pattern in data['patterns']:
        training_data.append(pattern.lower())
        labels.append(intent)

# Create and train the model
Vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words="english", max_df=0.8, min_df=1)
X_train = Vectorizer.fit_transform(training_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, labels, test_size=0.4, random_state=42, stratify=labels)

model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X_train, Y_train)

# Intent prediction function
def predict_intent(user_input):
    user_input = user_input.lower()
    input_vector = Vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]
    return intent

# Process user message and get response
def get_response(user_input):
    if user_input.lower() == 'exit':
        return "Goodbye!"
    
    # Special case for time intent
    if "time" in user_input.lower():
        time_now = datetime.now().strftime("%H:%M")
        return f"The current time is {time_now}."
    
    intent = predict_intent(user_input)
    if intent in intents:
        responses = intents[intent]['responses']        
        response = random.choice(responses)
        
        # Handle special placeholders in responses
        if "[current_time]" in response:
            current_time = datetime.now().strftime("%H:%M")
            response = response.replace("[current_time]", current_time)
        
        return response
    else:
        return "Sorry, I am not sure how to respond to that."

# Create a templates directory for the HTML files
os.makedirs('./templates', exist_ok=True)

# Write index.html to templates directory
with open('./templates/index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tyler Durden Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #c41e3a;
            --secondary-color: #2c2c2c;
            --text-color: #333;
            --light-bg: #f8f9fa;
            --dark-bg: #343a40;
            --message-bg: #e9ecef;
            --bot-message-bg: #f0f0f0;
            --border-radius: 10px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--light-bg);
            color: var(--text-color);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .container {
            max-width: 100%;
            height: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
        }

        .header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 1.8rem;
            margin-left: 10px;
        }

        .logo {
            width: 40px;
            height: 40px;
            margin-right: 10px;
        }

        .chat-container {
            flex-grow: 1;
            padding: 1rem;
            overflow-y: auto;
            background-color: white;
            border-radius: var(--border-radius);
            box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
            margin: 0 1rem;
        }

        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: var(--border-radius);
            max-width: 80%;
            word-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            position: relative;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            background-color: var(--primary-color);
            color: white;
            margin-left: auto;
            border-top-right-radius: 0;
        }

        .bot-message {
            background-color: var(--bot-message-bg);
            margin-right: auto;
            border-top-left-radius: 0;
        }

        .bot-message::before {
            content: '';
            position: absolute;
            width: 0;
            height: 0;
            top: 0;
            left: -10px;
            border-top: 10px solid var(--bot-message-bg);
            border-left: 10px solid transparent;
        }

        .user-message::after {
            content: '';
            position: absolute;
            width: 0;
            height: 0;
            top: 0;
            right: -10px;
            border-top: 10px solid var(--primary-color);
            border-right: 10px solid transparent;
        }

        .message-time {
            font-size: 0.7rem;
            opacity: 0.7;
            margin-top: 5px;
            text-align: right;
        }

        .input-container {
            display: flex;
            padding: 1rem;
            background-color: white;
            border-top: 1px solid #ddd;
        }

        .message-input {
            flex-grow: 1;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            outline: none;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .message-input:focus {
            border-color: var(--primary-color);
        }

        .send-button, .mic-button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            margin-left: 10px;
            padding: 0 15px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .send-button:hover, .mic-button:hover {
            background-color: #a01a2f;
        }

        .disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }

        .disabled:hover {
            background-color: #cccccc;
        }

        .listening {
            background-color: #28a745;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .message-status {
            font-size: 0.8rem;
            color: #666;
            text-align: center;
            margin: 5px 0;
            height: 20px;
        }

        .typing-indicator {
            display: inline-block;
            width: 50px;
            text-align: left;
        }

        .typing-indicator span {
            height: 8px;
            width: 8px;
            float: left;
            margin: 0 1px;
            background-color: #9E9EA1;
            display: block;
            border-radius: 50%;
            opacity: 0.4;
        }

        .typing-indicator span:nth-of-type(1) {
            animation: 1s blink infinite 0.3333s;
        }

        .typing-indicator span:nth-of-type(2) {
            animation: 1s blink infinite 0.6666s;
        }

        .typing-indicator span:nth-of-type(3) {
            animation: 1s blink infinite 0.9999s;
        }

        @keyframes blink {
            50% { opacity: 1; }
        }

        @media (max-width: 768px) {
            .message {
                max-width: 90%;
            }

            .header h1 {
                font-size: 1.5rem;
            }
        }

        @media (max-width: 480px) {
            .message {
                max-width: 95%;
            }

            .header h1 {
                font-size: 1.2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="/api/placeholder/40/40" alt="Tyler Durden Logo" class="logo">
            <h1>Tyler Durden Chatbot</h1>
        </div>
        
        <div class="chat-container" id="chat-container">
            <!-- Messages will be dynamically added here -->
            <div class="message bot-message">
                Hello! I'm Tyler Durden. How can I help you today?
                <div class="message-time">Just now</div>
            </div>
        </div>
        
        <div class="message-status" id="status-message"></div>
        
        <div class="input-container">
            <input type="text" class="message-input" id="message-input" placeholder="Type your message here..." autocomplete="off">
            <button class="mic-button" id="mic-button">
                <i class="fas fa-microphone"></i>
            </button>
            <button class="send-button" id="send-button">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chat-container');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');
            const micButton = document.getElementById('mic-button');
            const statusMessage = document.getElementById('status-message');
            
            // Check if speech recognition is available
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            let recognition;
            
            if (SpeechRecognition) {
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.lang = 'en-US';
                
                recognition.onstart = function() {
                    micButton.classList.add('listening');
                    updateStatus('Listening...');
                };
                
                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    messageInput.value = transcript;
                    updateStatus('Processing...');
                    sendMessage();
                };
                
                recognition.onerror = function(event) {
                    if (event.error === 'no-speech') {
                        updateStatus('No speech detected');
                    } else {
                        updateStatus('Error: ' + event.error);
                    }
                    micButton.classList.remove('listening');
                };
                
                recognition.onend = function() {
                    micButton.classList.remove('listening');
                };
                
                micButton.addEventListener('click', function() {
                    if (micButton.classList.contains('listening')) {
                        recognition.stop();
                    } else {
                        recognition.start();
                    }
                });
            } else {
                micButton.style.display = 'none';
                updateStatus('Speech recognition not supported in this browser');
            }
            
            // Handle send button click
            sendButton.addEventListener('click', sendMessage);
            
            // Handle Enter key press
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (message === '') return;
                
                // Add user message to chat
                addMessage(message, 'user');
                
                // Clear input
                messageInput.value = '';
                
                // Show typing indicator
                showTypingIndicator();
                
                // Send message to server
                fetch('/send_message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide typing indicator
                    hideTypingIndicator();
                    
                    // Add bot response to chat
                    setTimeout(() => {
                        addMessage(data.response, 'bot');
                    }, 500); // Slight delay to simulate thinking
                })
                .catch(error => {
                    hideTypingIndicator();
                    updateStatus('Error: Could not connect to server');
                    console.error('Error:', error);
                });
            }
            
            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
                
                messageDiv.textContent = text;
                
                const timeDiv = document.createElement('div');
                timeDiv.classList.add('message-time');
                timeDiv.textContent = getCurrentTime();
                messageDiv.appendChild(timeDiv);
                
                chatContainer.appendChild(messageDiv);
                
                // Scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                // If it's a bot message, consider text-to-speech
                if (sender === 'bot' && window.speechSynthesis) {
                    speakText(text);
                }
            }
            
            function showTypingIndicator() {
                statusMessage.innerHTML = `
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    Tyler is typing...
                `;
            }
            
            function hideTypingIndicator() {
                statusMessage.innerHTML = '';
            }
            
            function updateStatus(message) {
                statusMessage.textContent = message;
            }
            
            function getCurrentTime() {
                const now = new Date();
                let hours = now.getHours();
                let minutes = now.getMinutes();
                const ampm = hours >= 12 ? 'PM' : 'AM';
                
                hours = hours % 12;
                hours = hours ? hours : 12; // 0 should be 12
                minutes = minutes < 10 ? '0' + minutes : minutes;
                
                return hours + ':' + minutes + ' ' + ampm;
            }
            
            function speakText(text) {
                if (window.speechSynthesis) {
                    const utterance = new SpeechSynthesisUtterance(text);
                    
                    // Get voices
                    const voices = window.speechSynthesis.getVoices();
                    
                    // Try to find a male voice
                    let selectedVoice = voices.find(voice => voice.name.includes('Male') || voice.name.includes('David'));
                    
                    // If no specific male voice found, use the first available
                    if (!selectedVoice && voices.length > 0) {
                        selectedVoice = voices[0];
                    }
                    
                    if (selectedVoice) {
                        utterance.voice = selectedVoice;
                    }
                    
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;
                    
                    window.speechSynthesis.speak(utterance);
                }
            }
        });
    </script>
</body>
</html>""")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json['message']
    response = get_response(user_input)
    return jsonify({'response': response})

# For Vercel serverless function
@app.route('/<path:path>')
def catch_all(path):
    return home()

# Add route for speech recognition (if needed in the future)
@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    # This would be implemented with a JavaScript-based solution
    # since browser APIs handle speech recognition better than server-side
    pass

# Entry point for Vercel
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        with app.test_client() as test_client:
            response = test_client.get('/')
            self.wfile.write(response.data)

# For local development
if __name__ == '__main__':
    app.run(debug=True)
