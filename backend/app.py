from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')
model = BertForSequenceClassification.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')

# Define emotion labels
emotion_labels = ['anger', 'joy', 'optimism', 'sadness']

# Define depression keywords and responses (TO BE ADDED)
depression_keywords = {
   'hopeless': [
        "It’s hard when you feel hopeless. Do you want to talk more about it?",
        "Feeling hopeless can be overwhelming, but remember, you're not alone.",
        "You might feel stuck right now, but things can and will change.",
        "Sometimes, it's tough to see the light. How can I help you through this?",
        "You don’t have to navigate this feeling alone.",
        "It’s okay to feel hopeless sometimes. Let’s explore why you feel this way.",
        "Your emotions are valid. We can work through this together.",
        "It’s hard to feel hopeful in tough times. I’m here for you.",
        "Things may seem bleak, but change is possible. I believe in you.",
        "Feeling hopeless doesn’t define your worth. You’re important.",
        "What’s been on your mind that makes you feel hopeless?",
        "Even in hard times, there is hope. Let’s find it together.",
        "It’s okay to express these feelings. I’m here to listen.",
        "When everything feels bleak, small steps forward can help.",
        "Hopelessness can be heavy, but sharing it can ease the burden.",
        "You’re allowed to feel this way, but you don’t have to stay in it.",
        "Let’s explore what’s making you feel this way and work through it.",
        "This feeling is temporary. I’m here to help you through it.",
        "I hear you. I’m here to remind you that things can improve.",
        "You don’t have to face hopelessness alone. I’m here with you."
    ],
    'alone': [
        "I hear that you're feeling alone. Do you want to talk about what's on your mind?",
        "It's natural to feel alone sometimes. How are you coping with it?",
        "You don’t have to go through this alone. I’m here to support you.",
        "Loneliness can be tough. How can I be here for you?",
        "Reaching out is a step in the right direction. Let’s take this one step at a time.",
        "Being alone doesn’t mean you’re unloved. I’m here for you.",
        "I’m here to remind you that you are not truly alone.",
        "Feeling alone can be overwhelming. How can I support you?",
        "Loneliness is tough. Let’s find ways to make you feel more connected.",
        "You’re not invisible, and your feelings matter. I see you.",
        "Even when you feel alone, remember you are worthy of connection.",
        "What’s been making you feel isolated? Let’s talk through it.",
        "You may feel alone, but you don’t have to carry that weight on your own.",
        "I know it’s tough when you feel like no one understands. I’m here to listen.",
        "You’ve taken a brave step by sharing how you feel. Let’s talk.",
        "It’s okay to feel alone, but I’m here to offer you comfort and support.",
        "How can we make sure you feel less alone moving forward?",
        "Sometimes being alone is necessary for healing, but connection matters too.",
        "Loneliness doesn’t define who you are. You are valued.",
        "You don’t have to face this feeling by yourself. I’m here."
    ],
    'empty': [
        "Feeling empty is hard. Let’s take a moment and talk about it.",
        "Emptiness can feel so consuming. Do you want to share what’s going on?",
        "It's okay to feel this way. Let’s figure out what you need right now.",
        "I understand that you feel empty. Sometimes acknowledging it is the first step.",
        "It might feel like you're lacking something, but I'm here to help.",
        "Emptiness can feel overwhelming. I’m here to help you work through it.",
        "You’re not alone in feeling this way. Let’s unpack it together.",
        "What’s been making you feel empty lately? Let’s explore it.",
        "Emptiness doesn’t define your experience. We can fill it with care.",
        "You deserve to feel whole. Let’s work toward that together.",
        "Sometimes emptiness comes from exhaustion. Let’s find what recharges you.",
        "Even when you feel empty, your worth is still full and complete.",
        "It’s okay to not have all the answers right now. Let’s take it step by step.",
        "What can we do to make this feeling more manageable for you?",
        "I’m here to offer support, even when everything feels hollow.",
        "You don’t have to fill the emptiness alone. I’m here to help.",
        "It’s okay to feel this way. Let’s explore what’s beneath the surface.",
        "Even when you feel empty, there is still space for healing.",
        "You don’t have to carry this emptiness alone. I’m here for you.",
        "What’s missing in your life that’s making you feel empty?"
    ],
    'worthless': [
        "I’m really sorry you’re feeling worthless. You matter so much.",
        "Feeling worthless can be draining. Let’s work through this together.",
        "You are worthy, even if it doesn’t feel like it right now.",
        "You have value, and I’m here to remind you of that.",
        "It’s okay to feel this way, but remember, your worth isn’t determined by these feelings.",
        "You are more valuable than you realize. I’m here to help remind you.",
        "Your worth is inherent. Let’s talk about what’s been making you feel this way.",
        "It’s okay to feel this way, but let’s work through it together.",
        "You matter, even if you don’t feel it right now.",
        "Let’s explore what’s been weighing on you that makes you feel worthless.",
        "You are important. I’m here to remind you of that when it’s hard to see.",
        "Your worth isn’t tied to how you feel today. You have so much to offer.",
        "Even when you don’t feel it, you have intrinsic value.",
        "You’re not worthless. Your feelings are valid, but they don’t define you.",
        "Let’s talk about what’s been making you feel less than you truly are.",
        "You bring unique value to the world, even when you don’t feel it.",
        "I hear you’re feeling worthless, but I see your value clearly.",
        "It’s okay to feel this way, but know you have worth beyond this moment.",
        "You are worthy of love, care, and support. I’m here to remind you of that.",
        "I understand you feel this way, but let’s work together to shift this perspective."
    ],
    'tired': [
        "It sounds like you’re really tired. Do you want to talk about what's exhausting you?",
        "Exhaustion can take a toll, especially emotionally. What can we do to help you recharge?",
        "Feeling tired is understandable. Let’s explore ways to help you feel a little better.",
        "It’s okay to take a rest when you’re feeling tired.",
        "We can take this one step at a time. What’s making you feel this way?",
        "When you’re tired, it’s hard to stay positive. I’m here for you.",
        "You deserve rest. Let’s find ways to help you relax and recharge.",
        "I understand feeling tired. How can we lighten your load today?",
        "Tiredness can stem from many things. Let’s talk about it.",
        "It’s okay to take a break when everything feels overwhelming.",
        "You’ve been through a lot. It’s natural to feel tired. I’m here to help.",
        "Your body and mind need rest. How can I support you right now?",
        "It’s okay to take things slow when you’re feeling this way.",
        "When you’re tired, let’s focus on small steps to feel a bit better.",
        "You don’t have to do it all at once. Let’s break it down and take it easy.",
        "What can we do to make sure you’re taking care of yourself?",
        "I hear you. Feeling tired is tough. Let’s figure out how to restore your energy.",
        "You’ve been strong for so long. It’s okay to rest.",
        "Rest and care are important. Let’s work on ways to get you some relief.",
        "You’re not alone in feeling tired. Let’s figure out a plan to get you some rest."
    ],
    'overwhelmed': [
        "Feeling overwhelmed can be really tough. What’s weighing on you?",
        "It’s okay to feel overwhelmed. Let's try to break things down and figure it out together.",
        "Being overwhelmed can make everything feel like too much. How can I support you?",
        "Let’s take a breath and work through this one step at a time.",
        "Sometimes when things feel like too much, taking small steps can help.",
        "I know things feel heavy right now. How can I be there for you?",
        "It’s okay to feel overwhelmed. Let’s unpack it one piece at a time.",
        "You don’t have to solve everything at once. Let’s take it slow.",
        "I hear you. Feeling overwhelmed can be exhausting. Let’s break it down.",
        "You’re carrying a lot. Let’s figure out what we can lighten together.",
        "We can take small steps to help manage the things that are overwhelming you.",
        "When everything feels like too much, talking about it can help. I’m here.",
        "We don’t have to fix everything at once. Let’s start with what’s manageable.",
        "Feeling overwhelmed doesn’t mean you’re failing. It means you’re human.",
        "You don’t have to face all of this alone. I’m here for you.",
        "Let’s take a moment to breathe and assess what’s really going on.",
        "Being overwhelmed is tough, but we can find a way to navigate through it.",
        "You’ve been handling a lot. It’s okay to slow down and take a break.",
        "When life gets overwhelming, it’s okay to ask for help. I’m here.",
        "We’ll figure this out together. You don’t have to face it all at once."
    ],
    'sad': [
        "It’s okay to feel sad sometimes. I’m here for you.",
        "Feeling sad can be tough. Do you want to talk about it?",
        "I’m really sorry that you’re feeling sad. How can I support you?",
        "It’s normal to feel sad sometimes. What’s been on your mind?",
        "You don’t have to go through sadness alone. I’m here to listen.",
        "Sadness is valid. It’s okay to feel this way. I’m here for you.",
        "When you feel sad, it’s important to remember that things can get better.",
        "What’s been making you feel sad? Let’s talk through it.",
        "Your sadness matters, and I’m here to help you through it.",
        "Sometimes sadness can be overwhelming. I’m here for support.",
        "I’m sorry you’re feeling sad. I’m here if you want to talk more about it.",
        "It’s okay to feel down. You don’t have to hide your feelings.",
        "Let’s take some time to explore what’s been making you feel this way.",
        "Even when you’re feeling sad, remember that you’re not alone.",
        "You don’t have to face sadness on your own. Let’s talk about it.",
        "It’s okay to cry or express your sadness. I’m here for you.",
        "Sadness is a part of life, but you don’t have to carry it by yourself.",
        "You’re allowed to feel sad. How can I support you through this?",
        "Even in your sadness, you are valued and cared for.",
        "Let’s find a way to process this sadness together. I’m here."
    ],
   'isolated': [
        "I’m sorry you feel isolated. Let’s talk about what’s on your mind.",
        "Isolation can be so painful. I’m here to offer you support.",
        "It’s tough to feel isolated. Let’s explore how we can create connection.",
        "I understand that feeling isolated is hard. You don’t have to face it alone.",
        "Even in isolation, you’re not forgotten. I’m here for you.",
        "Isolation can feel suffocating, but reaching out is a positive step.",
        "Let’s talk about ways to help you feel less isolated.",
        "You don’t have to be isolated forever. There are ways to connect with others.",
        "What’s been making you feel isolated? Let’s work through it together.",
        "I’m here to listen and remind you that you are not alone.",
        "Isolation doesn’t define you. You are worth connection and love.",
        "It’s okay to feel isolated, but we can find ways to reconnect.",
        "When you feel isolated, know that reaching out for support is a strong move.",
        "You’re not alone in feeling isolated. I’m here with you.",
        "Let’s explore how we can make you feel more supported and less isolated.",
        "Isolation can feel heavy, but sharing it lightens the load. I’m here.",
        "You don’t have to face isolation by yourself. Let’s take this journey together.",
        "Even in your isolation, you are seen and valued. I’m here to support you.",
        "Let’s figure out what’s been making you feel so isolated and address it.",
        "You may feel isolated, but your experience matters. Let’s work through it."
    ],
    'worthless': [
        "Feeling worthless can make everything seem harder. I’m here to remind you of your value.",
        "You are worthy, no matter how you feel right now.",
        "I’m really sorry you’re feeling worthless, but your life matters to so many.",
        "You don’t have to feel this way alone. I’m here to support you.",
        "You have value, even if it’s hard to see right now. Let’s talk.",
        "Your worth is not based on how you feel today. You matter.",
        "What’s been making you feel worthless? I’m here to listen.",
        "Even when you feel worthless, I see your value clearly.",
        "You are important and worthy of love, care, and support.",
        "This feeling will pass, but your worth will remain. I’m here for you.",
        "Feeling worthless is painful, but you are not defined by these feelings.",
        "You’re worthy, and I’m here to remind you when it’s hard to remember.",
        "I know it’s tough, but I believe in your worth and want to help.",
        "You are not alone in feeling worthless, but it’s important to talk about it.",
        "Even when you can’t see your value, I see it. Let’s work through this.",
        "Your feelings are valid, but they don’t define your inherent worth.",
        "You don’t have to go through this feeling alone. I’m here to help.",
        "Let’s figure out what’s making you feel this way and work toward healing.",
        "You are not worthless. You bring so much to this world, even if you can’t see it.",
        "I’m here to remind you that you matter, no matter how worthless you feel."
    ],
    'numb': [
        "Feeling numb can be overwhelming. I’m here if you want to talk.",
        "Numbness is hard to deal with, but it’s okay to acknowledge it.",
        "You’re feeling numb, but I’m here to help you work through it.",
        "What’s been making you feel numb? Let’s talk about it.",
        "Numbness can be a way to protect ourselves from pain. Let’s explore this.",
        "It’s okay to feel numb. We can work through it together.",
        "You don’t have to face this numbness alone. I’m here to listen.",
        "Numbness doesn’t mean you’re broken. It’s just a part of the process.",
        "Sometimes feeling numb can make it hard to feel anything at all. I’m here for you.",
        "Let’s take small steps toward understanding this feeling of numbness.",
        "It’s okay to not feel anything right now. I’m here when you’re ready.",
        "You’re not alone in feeling numb. Let’s figure out what’s behind it.",
        "Numbness can feel isolating, but you’re not alone. I’m here for you.",
        "Let’s talk about what’s been making you feel numb recently.",
        "It’s tough to feel this way, but sharing it can help lighten the load.",
        "I understand that feeling numb can be scary. Let’s take it one step at a time.",
        "You may feel numb now, but emotions will return in time. I’m here to help.",
        "Numbness is a response to something deeper. Let’s explore it together.",
        "Even when you feel numb, you still matter. I’m here for you.",
        "It’s okay to not feel right now. I’ll be here as you navigate through this."
    ],
    'exhausted': [
        "Emotional exhaustion can be draining. How can I help support you?",
        "Feeling exhausted can make everything feel heavier. I’m here for you.",
        "It sounds like you’re really exhausted. Do you want to talk about it?",
        "Exhaustion is tough, but I’m here to offer my support.",
        "You’ve been through a lot. It’s okay to feel exhausted.",
        "Let’s take a moment to rest and talk about what’s been weighing on you.",
        "It’s okay to feel tired and drained. I’m here to listen if you want to share.",
        "Exhaustion can feel overwhelming. Let’s break things down together.",
        "You don’t have to do it all right now. It’s okay to rest.",
        "Let’s figure out what’s been making you feel exhausted and address it.",
        "You’ve been strong for a long time. It’s okay to take a break.",
        "Emotional exhaustion is real. Let’s talk about what’s been going on.",
        "I hear that you’re feeling exhausted. Let’s find ways to recharge.",
        "It’s tough to feel this way, but we can work through it together.",
        "Exhaustion can be heavy, but you don’t have to carry it alone.",
        "I’m here to support you through this exhaustion. Let’s talk about it.",
        "Even when you’re exhausted, you’re still strong. Let’s find a way to rest.",
        "You don’t have to push through this alone. I’m here to help lighten the load.",
        "Exhaustion can be overwhelming, but sharing it can help ease the burden.",
        "Let’s take some time to rest and recover. You deserve to feel better."
    ],
    'broken': [
        "I hear you’re feeling broken. Let’s talk about it.",
        "It’s okay to feel broken sometimes. I’m here to help you through it.",
        "You may feel broken, but you are still whole in so many ways.",
        "Even when you feel broken, there is hope for healing.",
        "You’re not broken beyond repair. We can work through this together.",
        "Feeling broken is tough, but you don’t have to navigate it alone.",
        "What’s been making you feel broken? Let’s explore it together.",
        "I’m sorry you feel this way. I’m here to offer my support.",
        "Even broken things can be mended. Let’s talk about how to start.",
        "You may feel broken now, but healing is always possible.",
        "You are not broken beyond help. I’m here to remind you of that.",
        "It’s okay to feel broken. Let’s work on putting the pieces back together.",
        "Feeling broken is valid, but you don’t have to stay in that place.",
        "Let’s explore what’s been making you feel this way and address it.",
        "I’m here to remind you that you are whole, even when you feel broken.",
        "You’re not alone in feeling broken. I’m here to support you.",
        "Brokenness is a part of life, but it’s not the end of the story.",
        "You may feel broken, but there is strength in your vulnerability.",
        "Let’s take small steps toward healing and mending what feels broken.",
        "Even when you feel broken, you are deserving of care and support."
    ],
    'fearful': [
        "It’s okay to feel afraid. Let’s talk about what’s been scaring you.",
        "Fear can be overwhelming, but I’m here to help you through it.",
        "What’s been making you feel fearful lately? Let’s work through it.",
        "It’s natural to feel fear sometimes. You’re not alone in this.",
        "Even when you’re afraid, I’m here to offer support and reassurance.",
        "Fear doesn’t have to control you. Let’s explore what’s behind it.",
        "You’re strong, even in moments of fear. I’m here to remind you of that.",
        "It’s okay to feel scared. I’m here to help you face it.",
        "Let’s figure out what’s been making you feel fearful and address it.",
        "You don’t have to face fear alone. I’m here to support you through it.",
        "Even when fear feels overwhelming, you are not alone in it.",
        "What’s been on your mind that’s making you feel fearful? Let’s talk.",
        "You may feel fear now, but we can work together to lessen it.",
        "I’m here to remind you that you are capable of handling what scares you.",
        "Let’s take small steps toward understanding and addressing this fear.",
        "It’s okay to feel fearful. I’m here to offer my support and guidance.",
        "You are stronger than your fears. Let’s figure out how to move forward.",
        "Fear is tough, but it doesn’t define you. Let’s work through it together.",
        "You’re not alone in your fear. I’m here to help you face it.",
        "Let’s talk about what’s been making you feel scared and find a way through."
    ],
    'isolated': [
        "I’m sorry you feel isolated. Let’s talk about what’s on your mind.",
        "Isolation can be so painful. I’m here to offer you support.",
        "It’s tough to feel isolated. Let’s explore how we can create connection.",
        "I understand that feeling isolated is hard. You don’t have to face it alone.",
        "Even in isolation, you’re not forgotten. I’m here for you.",
        "Isolation can feel suffocating, but reaching out is a positive step.",
        "Let’s talk about ways to help you feel less isolated.",
        "You don’t have to be isolated forever. There are ways to connect with others.",
        "What’s been making you feel isolated? Let’s work through it together.",
        "I’m here to listen and remind you that you are not alone.",
        "Isolation doesn’t define you. You are worth connection and love.",
        "It’s okay to feel isolated, but we can find ways to reconnect.",
        "When you feel isolated, know that reaching out for support is a strong move.",
        "You’re not alone in feeling isolated. I’m here with you.",
        "Let’s explore how we can make you feel more supported and less isolated.",
        "Isolation can feel heavy, but sharing it lightens the load. I’m here.",
        "You don’t have to face isolation by yourself. Let’s take this journey together.",
        "Even in your isolation, you are seen and valued. I’m here to support you.",
        "Let’s figure out what’s been making you feel so isolated and address it.",
        "You may feel isolated, but your experience matters. Let’s work through it."
    ],
    'lonely': [
        "I’m really sorry you’re feeling lonely. Let’s talk about what’s going on.",
        "It’s okay to feel lonely sometimes, but you’re not alone in this.",
        "Loneliness is tough, but I’m here to support you through it.",
        "Let’s figure out why you’re feeling lonely and work on it together.",
        "You’re not alone, even if it feels that way right now. I’m here.",
        "Feeling lonely can be overwhelming, but we’ll work through this together.",
        "What’s been making you feel lonely? Let’s talk about it.",
        "You deserve connection, even when loneliness feels like a barrier.",
        "Sometimes, sharing your loneliness can help lighten its weight.",
        "You don’t have to feel lonely forever. Let’s explore how to change that.",
        "Loneliness can feel consuming, but we’ll find ways to help you feel supported.",
        "Reaching out when you feel lonely is a sign of strength, not weakness.",
        "Your loneliness is real, but so is your capacity to connect with others.",
        "You don’t have to walk through this alone. Let’s work on it together.",
        "It’s hard to feel lonely, but I’m here to remind you that you’re cared for.",
        "You matter, and your loneliness doesn’t change that. Let’s talk.",
        "When loneliness hits, reaching out for support can make a world of difference.",
        "You deserve to feel connected. Let’s find ways to create those connections.",
        "I’m here to remind you that even in loneliness, you’re valued.",
        "Let’s figure out what’s been contributing to your loneliness and address it together."
    ],
    'withdrawn': [
        "I hear you’re feeling withdrawn. I’m here when you’re ready to talk.",
        "Withdrawing can be a way of coping. Let’s explore what’s going on.",
        "It’s okay if you’ve been feeling withdrawn. Take your time, I’m here for you.",
        "Sometimes withdrawing feels like the safest option, but you’re not alone.",
        "You don’t have to face everything alone. I’m here to listen whenever you’re ready.",
        "It’s hard to open up when you feel withdrawn. Let’s take it one step at a time.",
        "What’s been making you feel withdrawn? I’d love to help you through it.",
        "Being withdrawn doesn’t mean you’re alone. Let’s work on reconnecting.",
        "If you’ve been feeling withdrawn, that’s okay. Let’s talk when you’re ready.",
        "Your feelings are valid, even when you feel the need to withdraw.",
        "You don’t have to isolate yourself. I’m here to support you when you’re ready.",
        "Being withdrawn can feel safe, but I’m here to remind you that I care.",
        "Take your time if you’ve been feeling withdrawn. I’ll be here when you need me.",
        "Withdrawing doesn’t mean you’re weak. It’s okay to ask for help.",
        "Let’s explore what’s been making you feel withdrawn and find ways to address it.",
        "You don’t have to be withdrawn forever. We can take small steps together.",
        "It’s okay to take a break when you feel withdrawn, but know I’m always here.",
        "You’re not alone, even if you’ve been feeling withdrawn. I’m here to help.",
        "When you’re ready to share, I’ll be here to listen.",
        "Let’s work together to find ways to help you feel less withdrawn."
    ],
    'avoiding': [
        "It sounds like you’ve been avoiding things. Let’s talk about why.",
        "Avoiding things can be a response to stress. How can I help you with that?",
        "You might be avoiding something right now, but I’m here to support you.",
        "What’s been making you feel like you need to avoid things? Let’s talk.",
        "Avoidance can feel safe, but we can work through what’s bothering you.",
        "Avoiding things doesn’t solve the problem, but I’m here to help you tackle it.",
        "Let’s talk about why you’ve been avoiding things and how we can work on it.",
        "You don’t have to avoid everything alone. I’m here for you.",
        "It’s okay if you’ve been avoiding things. We can work through it together.",
        "Avoidance can be a coping mechanism, but I’m here to help you face things.",
        "You don’t have to avoid everything that’s making you anxious. Let’s work on it.",
        "Avoiding things might feel easier, but let’s explore ways to address them.",
        "What’s been making you avoid certain things? I’m here to help.",
        "Avoidance can make things feel worse. Let’s work on breaking that cycle.",
        "You don’t have to avoid difficult emotions. I’m here to help you process them.",
        "Let’s figure out why you’ve been avoiding things and take steps to overcome it.",
        "Avoiding things doesn’t mean you’re weak. I’m here to help you face them.",
        "It’s okay if you’ve been avoiding things. Let’s work on tackling them together.",
        "You might feel like avoiding things is the answer, but we can face them together.",
        "Let’s explore ways to handle what you’ve been avoiding, step by step."
    ],
    'suicide': [
        "I’m really sorry you’re feeling this way. Please reach out to someone who can help you immediately.",
        "You don’t have to face this alone. Please contact a mental health professional.",
        "I’m so sorry you’re feeling this way. Please, let someone close to you know.",
        "It’s so important to talk to someone who can help you right now.",
        "Please reach out to someone you trust or a professional for support.",
        "I’m really worried about you. Please get in touch with someone who can help.",
        "I’m sorry you’re feeling this way. There are people who care and want to help.",
        "It’s crucial to talk to someone who can support you through this.",
        "Please, you’re not alone. Reach out to a mental health professional right away.",
        "I’m deeply concerned for you. Please talk to someone who can help.",
        "These feelings are serious. Please get immediate help from a professional.",
        "I’m really sorry you’re feeling like this. Please reach out to someone who can provide support.",
        "You are not alone. Please talk to someone who can offer help right now.",
        "Your life is important. Please, seek help from a mental health professional.",
        "I’m very worried about you. Please talk to someone who can offer support.",
        "You don’t have to go through this alone. Please seek professional help.",
        "It’s crucial to reach out for help if you’re feeling this way.",
        "Please, reach out to a professional who can provide the help you need.",
        "These feelings matter, and you deserve support. Please reach out for help.",
        "I’m concerned for your safety. Please talk to someone who can help immediately."
    ],
    'kill': [
        "If you’re thinking of hurting yourself, please talk to someone immediately.",
        "It’s so important to reach out to someone who can help you right now.",
        "I’m really sorry you’re feeling this way. Please contact a mental health professional.",
        "Please get help immediately. Your safety is the most important thing.",
        "These feelings are serious. Please talk to someone who can help.",
        "It’s crucial to reach out to someone for support right now.",
        "You’re not alone, and help is available. Please talk to someone.",
        "Please reach out to someone you trust or a professional for support.",
        "I’m very worried about you. Please talk to a professional immediately.",
        "I’m really sorry you’re feeling like this. Please, let someone know.",
        "You are not alone in this. Please reach out to someone who can support you.",
        "I’m deeply concerned for you. Please contact someone who can help.",
        "Please, seek help immediately from someone who can offer support.",
        "Your life is so valuable. Please talk to someone who can help you.",
        "I’m worried for your safety. Please get in touch with someone who can help.",
        "These feelings matter, and you deserve support. Please reach out to someone.",
        "I’m really concerned for you. Please talk to someone who can help right now.",
        "Please contact a mental health professional. Help is available.",
        "Your safety is so important. Please talk to someone right away.",
        "I’m worried about you. Please get in touch with someone who can offer support."
    ],
    'hopeless': [
        "It’s tough to feel hopeless. What’s been weighing on you?",
        "I’m sorry you’re feeling this way. Let’s talk about what’s going on.",
        "Even when things feel hopeless, I’m here for you.",
        "You may feel stuck now, but there is always hope for change.",
        "Feeling hopeless doesn’t mean you are without options.",
        "Let’s figure out why things seem hopeless and how to move forward.",
        "You’re not alone in this feeling. Let’s take this one step at a time.",
        "It’s okay to feel hopeless sometimes, but I believe in your resilience.",
        "You might feel like there’s no way out, but we’ll find a path together.",
        "There is always a light, even if you can’t see it right now.",
        "Hopelessness is overwhelming, but I’m here to support you.",
        "It’s hard to see the future when you feel hopeless. Let’s work on it together.",
        "You matter, even when everything feels like too much.",
        "There are always options, even if they’re hard to see right now.",
        "Feeling hopeless is valid, but I believe in your strength to keep going.",
        "Even in hopeless moments, reaching out can help shift your perspective.",
        "You might feel stuck in hopelessness, but there are ways forward.",
        "You’re not alone in this. Let’s find a way to move through the hopelessness.",
        "Even when things feel bleak, I’m here to remind you that hope exists.",
        "Let’s work together to find some light, even when everything feels dark."
    ],
    'helpless': [
        "I’m sorry you’re feeling helpless. Let’s talk about what’s going on.",
        "Feeling helpless is overwhelming, but we can find ways to cope together.",
        "It’s tough to feel helpless, but you don’t have to face this alone.",
        "Even when things feel out of your control, there are options to explore.",
        "You might feel helpless now, but there are ways forward.",
        "I understand that helplessness can feel crushing, but we can work through it.",
        "Let’s talk about why you’re feeling helpless and how we can address it.",
        "You don’t have to be helpless forever. Let’s find some solutions together.",
        "Feeling helpless doesn’t mean you are powerless. Let’s work on it together.",
        "There are ways to regain a sense of control, even when things feel overwhelming.",
        "You might feel helpless, but I’m here to remind you of your strength.",
        "We can work on finding solutions, even when you feel helpless.",
        "You don’t have to do this alone. I’m here to support you through this.",
        "Let’s figure out what’s contributing to your sense of helplessness and address it.",
        "Feeling helpless can be suffocating, but we’ll work through it together.",
        "You are not helpless, even if it feels like that right now.",
        "There are always ways to regain a sense of control. Let’s explore them.",
        "You might feel like things are out of your hands, but there is hope.",
        "Even in helpless moments, reaching out can make a difference.",
        "You’re not alone in this. Let’s work together to find some solutions."
    ],
    'end': [
        "I’m sorry you’re feeling like things should end. Please reach out for help.",
        "It’s so important to talk to someone who can support you right now.",
        "These feelings are really serious. Please reach out to someone who can help.",
        "If you’re feeling like things should end, please contact a mental health professional.",
        "You don’t have to face these feelings alone. Help is available.",
        "Please reach out to someone close to you or a professional for support.",
        "I’m really worried about you. Please get in touch with someone who can help.",
        "You might feel like things should end, but there is help available.",
        "Please reach out to someone who can provide the support you need.",
        "It’s important to talk to someone if you’re feeling like things should end.",
        "I’m deeply concerned for your well-being. Please contact someone who can help.",
        "You don’t have to go through this alone. Please seek help.",
        "These feelings are heavy, but you deserve support. Please reach out to someone.",
        "Please talk to someone who can help you through this difficult time.",
        "It’s crucial to reach out for help when you’re feeling like sthings should end.",
        "You are not alone in this. Please reach out for help.",
        "Please seek professional help if you’re feeling like things should end.",
        "I’m really sorry you’re feeling this way. Let someone know what’s going on.",
        "You don’t have to end things. There is help and support available.",
        "Please talk to someone who can provide you with support and care."
    ]
}

def predict_emotion(text):
    # Tokenize input text and prepare for model
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get model predictions
    outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get predicted emotion class
    predicted_class = torch.argmax(probs).item()
    
    # Return emotion label corresponding to predicted class
    return emotion_labels[predicted_class]

def get_response(emotion):
    # Generate response based on predicted emotion
    if emotion == 'anger':
        return "I sense some anger. It’s okay to feel this way. Would you like to talk more about what's bothering you?"
    elif emotion == 'joy':
        return "I’m glad you’re feeling joyful! It’s great to celebrate positive emotions. How can I support you today?"
    elif emotion == 'optimism':
        return "That’s wonderful! Optimism is such a powerful mindset. Keep up the positive thinking!"
    elif emotion == 'sadness':
        return "I'm really sorry you're feeling sad. It’s okay to feel this way. I'm here for you. Do you want to talk more about it?"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user message from JSON request
        data = request.get_json()
        user_message = data['message'].lower()
        
        # Check if message contains depression-related keyword
        for keyword, responses in depression_keywords.items():
            if keyword in user_message:
                # Randomly select response
                response = responses[torch.randint(0, len(responses), (1,)).item()]
                return jsonify({'keyword': keyword, 'response': response})
        
        # Predict emotion and respond
        emotion = predict_emotion(user_message)
        response = get_response(emotion)
        return jsonify({'emotion': emotion, 'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)