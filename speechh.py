import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import pyttsx3
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
import google.generativeai as genai
import pandas as pd
from datetime import datetime
api_key=st.secrets["bilal_api"]

# Configure Gemini API key
genai.configure(api_key=api_key)  # 🔁 Replace with your actual API key

# Download necessary NLTK resources
nltk.download("punkt")

# Text-to-speech function
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Speech-to-text function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak_text("Please speak when you're ready...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        speak_text("I didn't catch that. Could you say it again?")
        return ""
    except sr.RequestError:
        speak_text("There’s an issue with the speech recognition service.")
        return ""

# Ask Gemini for a response
def ask_genai(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

# Nickname introduction
def get_nickname():
    speak_text("Welcome to our session. What is your name?")
    name = speech_to_text()

    speak_text("Do you prefer me to call you by your name or a nickname?")
    response = speech_to_text()

    if response.lower() in ['nickname', 'yes', 'y']:
        speak_text("What nickname would you like me to use?")
        nickname = speech_to_text()
    else:
        nickname = name

    speak_text(f"Thank you. I’ll call you {nickname}. Let’s get started.")
    return nickname

# Ask questions and get GenAI feedback — ✨ Updated for adaptive follow-ups
def ask_questions(nickname):
    questions = [
        "Can you tell me how your day has been so far?",
        "What is something you enjoy doing?",
        "How do you feel when you spend time with your family or friends?",
        "What is one thing that makes you happy?",
        "How do you feel when trying something new or unexpected?"
    ]

    polarity_sum = 0
    responses = {}

    for i, question in enumerate(questions, 1):
        speak_text(question)
        response = speech_to_text()
        responses[f"Question {i}"] = response

        if response:
            sentiment = TextBlob(response)
            polarity_sum += sentiment.polarity

            # Instant GenAI feedback
            genai_prompt = f"The child said: '{response}'. Give a short, supportive reply in 1 sentence."
            genai_response = ask_genai(genai_prompt)
            speak_text(genai_response)

            # ✨ Adaptive follow-up question based on the response
            followup_prompt = f"""
You are a child therapist helping an autistic child in a friendly talk. Based on this response:
"{response}"
Ask a simple, kind follow-up question that helps the child express a little more, in a gentle and non-technical way. The question should be suitable for a child and continue the conversation naturally. Respond with just one question.
"""
            followup_question = ask_genai(followup_prompt)
            speak_text(followup_question)
            followup_response = speech_to_text()

            if followup_response:
                responses[f"Follow-up to Question {i}"] = followup_response
                sentiment = TextBlob(followup_response)
                polarity_sum += sentiment.polarity

                # Optional feedback for follow-up too
                genai_followup_reply = ask_genai(f"The child replied: '{followup_response}'. Give a gentle, 1-sentence reply.")
                speak_text(genai_followup_reply)

    for question, answer in responses.items():
        st.write(f"**{question}**: {answer}")

    return responses, polarity_sum

# Final mood interpretation
def interpret_score(score):
    if score > 2.5:
        return "You seem to be in a positive and cheerful mood overall. That's wonderful!"
    elif score >= 0.5:
        return "You appear to be feeling okay, with some positive moments. It’s important to acknowledge those feelings."
    elif score > -0.5:
        return "It seems you might be feeling neutral or a mix of emotions. That’s perfectly okay. Talking more about it could help."
    else:
        return "You might be experiencing some challenges or negative feelings. I’m here to listen and support you."

# Type-Token Ratio (TTR)
def calculate_ttr(text):
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0

# Repeated words
def detect_repeated_words(text):
    tokens = word_tokenize(text.lower())
    freq_dist = FreqDist(tokens)
    return {word: count for word, count in freq_dist.items() if count > 1}

# Overall sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# === Streamlit UI ===
st.title("🧠 Child Emotion & Language Interaction System")
st.markdown("This app analyzes a child’s spoken emotional expression using speech recognition, sentiment analysis, and Gemini AI feedback.")

if st.button("🎤 Start Interaction"):
    nickname = get_nickname()
    responses, polarity_score = ask_questions(nickname)
    combined_responses = " ".join(responses.values())

    ttr = calculate_ttr(combined_responses)
    repeated_words = detect_repeated_words(combined_responses)
    polarity, subjectivity = analyze_sentiment(combined_responses)
    mood = interpret_score(polarity_score)

    # Summary to Gemini
    summary_prompt = f"""
You are an experienced child therapist specializing in working with autistic children. You are analyzing a recent therapeutic interaction based on the child's spoken responses and language-use metrics.

Here is the child’s full response transcript:
"{combined_responses}"

Additional analysis from the session:
- Vocabulary Diversity (Type-Token Ratio): {ttr:.2f}
- Repeated Words (and their counts): {repeated_words}
- Overall Mood Assessment: {mood}
- Combined Sentiment Polarity Score: {polarity_score:.2f}
- Sentiment Polarity (TextBlob): {polarity:.2f}
- Subjectivity (TextBlob): {subjectivity:.2f}

Your task:
1. Gently assess the child’s emotional and mental state based on their responses and sentiment scores. Highlight any patterns in their feelings or tone.
2. Evaluate their communication style using the Type-Token Ratio and repeated words — identify if the child is expressive, repetitive, or reserved, and interpret what that suggests about their comfort and cognition.
3. Based on the emotional state and language style, provide a short, encouraging psychological interpretation in a way that helps the child feel understood and validated.
4. End by suggesting **one small therapeutic activity**, calming affirmation, or social-emotional learning tip that would be helpful for this child based on the mood, expression, and repetition patterns.

Please keep the tone nurturing, supportive, and age-appropriate. Avoid technical terms. Write as if you’re giving spoken feedback in a calm and friendly therapist voice.
Respond with a 4–5 sentence summary followed by 1–2 sentence advice or activity for the child.

You are a language development specialist working with autistic children. Based on the following session data, provide a short, friendly lesson or tip to help the child improve their communication. Focus on vocabulary diversity, reducing repeated words, or grammar — depending on what's most relevant.

Session details:
- Child's spoken transcript: "{combined_responses}"
- Vocabulary Diversity (TTR): {ttr:.2f}
- Repeated Words: {repeated_words}
- Sentiment Polarity: {polarity:.2f}
- Subjectivity: {subjectivity:.2f}

Your response should be 2–4 sentences long, simple, and age-appropriate. Use positive encouragement and avoid technical jargon. Conclude with one small exercise or suggestion the child can try to practice better language expression.
"""

    final_feedback = ask_genai(summary_prompt)

    # Create data record
    data_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nickname": nickname,
        "Combined_Sentiment_Score": polarity_score,
        "Vocabulary_Diversity_TTR": ttr,
        "Repeated_Words": str(repeated_words),
        "Overall_Mood": mood,
        "Sentiment_Polarity": polarity,
        "Subjectivity": subjectivity,
        "Full_Responses": combined_responses,
        "Therapist_Feedback": final_feedback
    }

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame([data_record])
    csv_filename = "child_interaction_data.csv"
    
    try:
        existing_df = pd.read_csv(csv_filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        updated_df = df
    
    updated_df.to_csv(csv_filename, index=False)

    st.subheader(f"🗣️ Feedback for {nickname}")
    st.write(final_feedback)
    speak_text(f"Thanks for sharing, {nickname}. Here’s what I learned:")
    speak_text(final_feedback)

    # Download buttons
    st.subheader("📥 Download Session Data")
    
    with open(csv_filename, "rb") as f:
        st.download_button(
            label="Download Full Session History (CSV)",
            data=f,
            file_name=csv_filename,
            mime="text/csv"
        )
    
    st.download_button(
        label="Download This Session's Feedback (TXT)",
        data=final_feedback,
        file_name=f"{nickname}_feedback_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
