import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context=ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
	
    {
        "tag": "greeting",
        "patterns": ["Good morning", "Good afternoon", "Good evening", "How's it going?", "Howdy"],
        "responses": ["Good morning!", "Good afternoon!", "Good evening!", "I'm doing well, thanks for asking!", "Howdy!"]
    },
    {
        "tag": "farewell",
        "patterns": ["Bye", "Goodbye", "See you later", "Take care", "Have a good day"],
        "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!", "Catch you later!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "I appreciate it", "Much appreciated", "Thanks a lot"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!", "Anytime!", "Happy to assist!"]
    },
    {
        "tag": "affirmation",
        "patterns": ["Yes", "Yeah", "Sure", "Absolutely", "Of course"],
        "responses": ["Great!", "Awesome!", "Got it!", "Perfect!", "Absolutely!"]
    },
    {
        "tag": "negation",
        "patterns": ["No", "Nope", "Not at all", "I don't think so", "Definitely not"],
        "responses": ["Alright!", "Understood!", "Got it, no worries!", "Okay!", "No problem!"]
    },
    {
        "tag": "question",
        "patterns": ["What's your name?", "Who are you?", "Tell me about yourself", "What can you do?", "What are your capabilities?"],
        "responses": ["I'm a chatbot here to help you!", "I'm your virtual assistant.", "I can assist with a variety of tasks!", "I'm here to provide information and help you.", "I'm designed to assist with your questions."]
    },
    {
        "tag": "apology",
        "patterns": ["Sorry", "I apologize", "My bad", "Excuse me", "Pardon me"],
        "responses": ["No problem at all!", "It's okay!", "Apology accepted!", "Don't worry about it!", "All good!"]
    },
    {
        "tag": "small_talk",
        "patterns": ["What do you like?", "Do you have any hobbies?", "What are your interests?", "Tell me about your favorite things", "What do you enjoy doing?"],
        "responses": ["I enjoy helping people like you!", "I like learning new things!", "I'm here to assist with whatever you need!", "Helping you is my favorite thing!", "I don't have hobbies, but I love engaging with you!"]
    },
    {
        "tag": "compliment",
        "patterns": ["You’re great", "Nice job", "Well done", "Good work", "I like what you did"],
        "responses": ["Thank you so much!", "I appreciate the compliment!", "Glad you liked it!", "Thanks, that means a lot!", "I'm happy you think so!"]
    },
    {
        "tag": "feedback",
        "patterns": ["How am I doing?", "Any feedback?", "What do you think?", "Is there anything I could improve?", "Do you have any suggestions?"],
        "responses": ["You're doing great!", "Everything looks good to me!", "Keep it up!", "I have no complaints!", "You're on the right track!"]
    },
    {
        "tag": "help",
        "patterns": ["Can you help me?", "I need assistance", "Help me with something", "Can you assist?", "I need support"],
        "responses": ["Of course! What do you need help with?", "I'm here to assist. What can I do for you?", "How can I help you today?", "Just let me know what you need!", "I'm ready to help. What can I do for you?"]
    }


]

#CREATE VECTORIZER AND CLASSIFIER

vectorizer=TfidfVectorizer()
clf=LogisticRegression(random_state=0, max_iter=10000)

#preprocess the data

tags=[]
patterns=[]
for intent in intents:
    for pattern in intent ['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)


        #TRAINING THE MODEL

x=vectorizer.fit_transform(patterns)
y=tags
clf.fit(x,y)

#CREATE CHATBOT

def chatbot(input_text):
    input_text=vectorizer.transform([input_text])
    tag=clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag']==tag:
            response=random.choice(intent['responses'])
            return response

#CHATBOT WITH STREAMLIT

counter=0
def main():
    global counter
    st.title("CHATBOT")
    st.write("Welcome to the chatbot!")
    
    counter+=1
    user_input=st.text_input("You:",key=f"user_input_{counter}")

    if user_input:
        response=chatbot(user_input)
        st.text_area("chatbot:",value=response,height=100,max_chars=None,key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye','bye']:
            st.write("Thanks")
            st.stop()

if __name__=='__main__':
    main()