import io
import os
import corpora
import pickle
import random
import string
import threading
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from textblob import TextBlob
from playsound import playsound
from fuzzywuzzy import process
from gtts import gTTS
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import streamlit as st
import subprocess
import sys
subprocess.run([f"{sys.executable}", "script.py"])
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
# Google Calendar API scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.events.readonly',
]

# Predefined responses dictionary
predefined_responses = {
    "Hello": "How can I help you?",
    "what are the appointment availability days?": "Consultation: Mondays, Wednesdays, Fridays - 9:00 AM to 12:00 PM. Routine Check-Up: Tuesdays and Thursdays - 2:00 PM to 5:00 PM. Emergency Visit: Daily availability with open slots between 8:00 AM to 9:00 AM, 1:00 PM to 2:00 PM, and 5:00 PM to 6:00 PM.",
    "I want to book an appointment": "There are multiple slots available, Daily availability with open slots between 8:00 AM to 9:00 AM, 1:00 PM to 2:00 PM, and 5:00 PM to 6:00 PM. Please let us know your suitable slot",
    "what types of appointments can I book?": "Appointment Types: Initial Consultation for new patients, Follow-Up for ongoing check-ups, and Teeth Whitening Trial for a 30-minute session.",
    "what is the rescheduling policy?": "Patients may reschedule appointments up to 24 hours in advance without any penalty.",
    "what is the cancellation policy?": "Cancellations within 24 hours of the appointment may incur a 50% service fee unless it's an emergency.",
    "what is the no-show policy?": "Patients who do not show up without notice may be charged the full consultation fee.",
    "how often should I come in for a check-up?": "We recommend coming in every six months for a routine check-up to maintain good oral health.",
    "what should I do in case of a dental emergency?": "If you experience pain, swelling, or bleeding, please contact our clinic immediately or come in for an emergency appointment.",
    "how long does teeth whitening take?": "In-office teeth whitening typically takes about 1-1.5 hours, depending on the level of whitening desired.",
    "how long does a teeth cleaning take?": "A routine cleaning takes about 45 minutes and includes plaque removal, polishing, and flossing. Avoid eating or drinking for 30 minutes after the cleaning for best results.",
    "how long does a root canal treatment take?": "A root canal treatment typically takes 1-2 hours. You may experience soreness afterward; avoid chewing with the affected tooth for a few days.",
    "what does teeth whitening involve?": "Teeth whitening uses a peroxide-based gel applied under a UV light to whiten teeth. Avoid staining foods and beverages for 24 hours post-treatment.",
    "what are the clinic hours?": "Our clinic hours are Monday to Friday - 8:00 AM to 6:00 PM; Saturday - 8:00 AM to 2:00 PM; Closed on Sundays.",
    "where is the clinic located?": "Our location is 123 Dental Way, Suite 456, Medical District, Springfield.",
    "what is the clinic contact number?": "You can contact us at (123) 456-7890 or email us at info@springfielddental.com.",
    "what are the age requirements for teeth whitening?": "Patients must be at least 18 years old for teeth whitening or dental implants.",
    "what medical history is required for dental procedures?": "No current infections or bleeding disorders; patients with a history of severe allergies need medical clearance.",
    "do I need a dental check-up before a cosmetic procedure?": "Patients who haven't had a recent check-up may be advised to schedule one first.",
    "can I have anesthesia for dental procedures?": "Patients without allergies to anesthesia are cleared for most dental procedures.",
    "who is the root canal specialist?": "Dr. Alice Thompson is our Root Canal Specialist, handling all root canal treatments and related inquiries.",
    "who is the cosmetic dentistry specialist?": "Dr. John Ellis is our Cosmetic Dentistry Specialist, managing teeth whitening, veneers, and other aesthetic procedures.",
    "who is the oral surgeon?": "Dr. Emily Wong is our Oral Surgeon, available for extractions, dental implants, and other surgical treatments.",
    "what should I do if I have ongoing pain after a root canal?": "Please schedule an immediate follow-up with Dr. Thompson, our Root Canal Specialist.",
    "who should I contact about dental implants?": "For questions about potential risks and aftercare for dental implants, please contact Dr. Wong, the Oral Surgeon."
}

# Preprocessing functions
def LemNormalize(text):
    # Normalize text using TextBlob for lemmatization
    blob = TextBlob(text)
    return [word.lemmatize() for word in blob.words]

# Tokenize input corpus for fallback
with open('intro_join.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()
sent_tokens = [str(sentence) for sentence in TextBlob(raw).sentences]

# Main response function
def response(user_response):
    user_response = user_response.lower().strip()

    # Attempt to match with predefined responses
    match, score = process.extractOne(user_response, predefined_responses.keys())
    if score > 80:
        return predefined_responses[match]

    # Check specific time-based conditions
    if "book an appointment" in user_response:
        return "There are multiple slots available: 8:00 AM - 9:00 AM, 1:00 PM - 2:00 PM, 5:00 PM - 6:00 PM."

    appointment_time_dict = {
        "10:00 p.m.": datetime.now().replace(hour=22, minute=0, second=0),
        "1:00 p.m.": datetime.now().replace(hour=13, minute=0, second=0),
        "5:00 p.m.": datetime.now().replace(hour=17, minute=0, second=0),
    }

    for time_str, start_time in appointment_time_dict.items():
        if time_str in user_response:
            end_time = start_time + timedelta(hours=1)
            create_appointment(start_time, end_time, 'ik224118@gmail.com', 'ismaeel6034@gmail.com')
            return f"OK, your appointment has been booked for {time_str}. Please make sure you're available."

    # Enhanced fallback response using TF-IDF similarity
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1, 2))
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Calculate cosine similarity and find the best match
    cosine_vals = cosine_similarity(tfidf[-1], tfidf)
    similarity_scores = cosine_vals.flatten()[:-1]
    best_match_idx = similarity_scores.argsort()[-1]
    best_match_score = similarity_scores[best_match_idx]
    
    # Threshold to filter out weak matches
    similarity_threshold = 0.3  # Adjust for stricter or looser matching
    
    # Ensure that a valid response is chosen
    if best_match_score < similarity_threshold:
        return "I am sorry! I don't understand you."
    else:
        return sent_tokens[best_match_idx]

# Speech recognition setup
r = sr.Recognizer()

def authenticate_google_calendar():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def create_appointment(start_time, end_time, patient_email, clinic_email):
    creds = authenticate_google_calendar()
    service = build('calendar', 'v3', credentials=creds)

    event = {
        'summary': 'Dental Appointment',
        'location': '123 Dental Way, Suite 456, Medical District, Springfield',
        'description': 'Dental appointment for patient.',
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/New_York',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'America/New_York',
        },
        'attendees': [
            {'email': patient_email},
            {'email': clinic_email},
        ],
    }

    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f'Appointment created: {event.get("htmlLink")}')
    return event.get('htmlLink')

# Function to handle voice response
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Main chat loop
def chatbot():
    print("Chatbot is listening...")
    while True:
        with sr.Microphone() as source:
            audio = r.listen(source)
            try:
                user_input = r.recognize_google(audio)
                print("User:", user_input)
                bot_response = response(user_input)
                print("Bot:", bot_response)
                speak(bot_response)
            except sr.UnknownValueError:
                print("Sorry, I did not catch that.")
                speak("Sorry, I did not catch that.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

if __name__ == "__main__":
    threading.Thread(target=chatbot).start()
