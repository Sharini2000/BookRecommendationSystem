from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import book_recommendation as br  # Fix: Import module properly
import logging


logging.basicConfig(level=logging.INFO)

# Flask app initialization
app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app)

# HTTP webhook route for handling POST requests
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    logging.info(f"Webhook Request: {req}")

    intent_name = req.get("queryResult", {}).get("intent", {}).get("displayName")
    book_title = req.get("queryResult", {}).get("parameters", {}).get("booktitle")

    if not book_title:
        return jsonify({"fulfillmentText": "Please provide a book title."})

    response_text = "Unknown request."
    try:
        if intent_name == "General Information":
            response_text = br.bookInformation(book_title)
        elif intent_name == "PositiveFeedback":
            response_text = br.generate_feedback(book_title, 'positive')
        elif intent_name == "NegativeFeedback":
            response_text = br.generate_feedback(book_title, 'negative')
        elif intent_name == "SimilarBooksEntites":
            response_text = str(br.get_recommendations_with_entities(book_title))
        elif intent_name == "ReviewTopics":
            response_text = br.provide_concise_feedback(book_title)
    except Exception as e:
        logging.error(f"Error handling intent '{intent_name}': {str(e)}")
        response_text = "Sorry, we might not have the book you are looking for. Some of the books which details we have are afterworld, bell toll, love dog, magician tale, vampire story and more."

    return jsonify({"fulfillmentText": response_text})

# WebSocket event handling
@socketio.on('connect')
def handle_connect():
    logging.info("Client connected")
    emit("response", {"message": "Connection established!"})

@socketio.on('message')
def handle_message(message):
    logging.info(f"Received message: {message}")

    if not isinstance(message, dict):
        emit("response", {"message": "Invalid message format."})
        return

    book_title = message.get("book_title", "")
    intent_name = message.get("intent", "")

    response_text = "Unknown request."
    if intent_name == "General Information":
        response_text = br.bookInformation(book_title)
    elif intent_name == "PositiveFeedback":
        response_text = br.generate_feedback(book_title, 'positive')
    elif intent_name == "NegativeFeedback":
        response_text = br.generate_feedback(book_title, 'negative')
    elif intent_name == "SimilarBooksEntites":
        recommendations = str(br.get_recommendations_with_entities(book_title))
        response_text = ", ".join(recommendations)
    elif intent_name == "ReviewTopics":
        response_text = br.provide_concise_feedback(book_title)

    emit('response', {"message": response_text})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)



