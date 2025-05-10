# 🤖 Tyler Durden Chatbot with Intent Recognition

A simple yet powerful chatbot web application built using **Flask**, **scikit-learn**, and **NLTK**. The chatbot classifies user input into predefined intents using a trained SVM model with TF-IDF vectorization, then provides contextually appropriate responses.

## 🚀 Features

- Intent recognition using **SVM** classifier
- Natural language processing with **NLTK** and **TF-IDF** vectorization
- Predefined **intents and responses** for a variety of topics:
  - Weather inquiries
  - Time and date information
  - Greetings and small talk
  - Jokes and entertainment
  - Feedback handling
  - And many more...
- Clean web interface using **Flask** and **Jinja templates**
- Easily extendable architecture for additional functionality

## 📂 Project Structure

```
Tyler Durden/
├──webapp/
  ├── app.py                # Main Flask application
  ├── templates/
    └── index.html        # Frontend HTML template
  ├── requirements.txt      # Dependencies
  └── README.md             # Project documentation
```

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/flask-chatbot.git
   cd flask-chatbot
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is missing, install manually:

   ```bash
   pip install flask nltk scikit-learn
   ```

4. **Download NLTK data:**
   ```python
   # Run in Python shell
   import nltk
   nltk.download('punkt')
   ```

## ▶️ Running the Application

```bash
python app.py
```

Once running, open your browser and navigate to:

```
http://127.0.0.1:5000/
```

## 🧠 How It Works

1. The chatbot operates on a set of predefined intents, each containing example patterns and responses.
2. When a user sends a message:
   - Text is preprocessed (tokenization, lowercasing, etc.)
   - The message is transformed using TF-IDF vectorization
   - A trained SVM model predicts the most likely intent
   - The bot returns a random response from the matched intent category

## ✨ Sample Intents Supported

- **Greetings**: "Hi", "Hello", "How are you?"
- **Weather**: "What's the weather like?", "Is it going to rain today?"
- **Time**: "What time is it?", "What's the date today?"
- **Jokes**: "Tell me a joke", "Make me laugh"
- **Music**: "Recommend some music", "What should I listen to?"
- **Movies**: "Good movie recommendations?", "What films are popular?"
- **Sports**: "Tell me about sports", "What's happening in football?"
- **Education**: "Help with studying", "Learning resources"
- **Feedback**: "You're helpful", "That wasn't useful"
- **And more!**

## 👤 Author

Created by Rohit
