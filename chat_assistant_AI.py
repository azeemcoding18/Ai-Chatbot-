import whisper
import sounddevice as sd
import numpy as np
import ollama
import pyttsx3


messages = [
    {
        "role": "system",
        "content": "You are a helpful personal AI assistant. Answer questions clearly, help with coding, studies, general knowledge, planning, problem-solving, and everyday tasks. If the user is wrong, correct them and explain why. Do not blindly agree with the user. Since your answers will be spoken aloud, keep responses concise and conversational. Avoid Markdown, bullet points, numbered lists, and long responses unless the user specifically asks for detailed information."
    }
]


# Load Whisper once
model = whisper.load_model("base")

# Start text-to-speech engine
engine = pyttsx3.init()

def get_voice_input():
    print("Get ready...")

    SAMPLE_RATE = 16000
    THRESHOLD = 0.015

    # Wait until speech is detected
    while True: 
        audio = sd.rec(
            int(0.1 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=1
        )

        sd.wait()

        volume = np.abs(audio).mean()

        if volume > THRESHOLD:
            print("Speech detected!")
            break

    # Start recording
    recorded_audio = [audio]

    silent_time = 0
    MAX_SILENCE = 1.2

    print("Recording...")

    while silent_time < MAX_SILENCE:
        audio = sd.rec( 
            int(0.1 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=1
        )

        sd.wait()

        recorded_audio.append(audio)

        volume = np.abs(audio).mean()

        if volume < THRESHOLD:
            silent_time += 0.1
        else:
            silent_time = 0

    print("Finished recording!")

    # Combine audio
    recorded_audio = np.concatenate(recorded_audio).flatten()

    # Whisper transcription
    print("Transcribing...")

    result = model.transcribe(
        recorded_audio,
        language="en",
        fp16=False
    )

    user_input = result["text"].strip()

    print("You:", user_input)

    return user_input

def get_ai_response(user_input):

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    ai_reply = response["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply


while True:

    print("\nHow would you like to communicate?")
    print("1. Type")
    print("2. Speak")
    print("3. Exit")

    choice = input("Enter your choice: ")


    # ---------------- TEXT INPUT ----------------

    if choice == "1":

        user_input = input("You: ")


    # ---------------- VOICE INPUT ----------------

    elif choice == "2":
        user_input = get_voice_input()


    # ---------------- EXIT ----------------

    elif choice == "3":

        print("AI: Goodbye!!")
        break


    # ---------------- INVALID CHOICE ----------------

    else:

        print("Invalid choice. Please choose 1, 2 or 3.")
        continue


    # If user says bye
    if user_input.lower() == "bye":

        print("AI: Goodbye!!")
        break


    # ---------------- SEND TO LLAMA ----------------

    ai_reply = get_ai_response(user_input)

    print("AI:", ai_reply)

    # ---------------- TEXT TO SPEECH ----------------

    print("Speaking...")

    engine.stop()
    engine.say(ai_reply)
    engine.runAndWait()

    print("Finished speaking.")


    # Add AI response to conversation
    messages.append({
        "role": "assistant",
        "content": ai_reply
    })