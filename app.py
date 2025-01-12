import streamlit as st
from google.cloud import speech
import moviepy.editor as mp
import io
import os
from google.cloud import texttospeech
from pydub import AudioSegment
import requests
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Provide the path to your service account key JSON
SERVICE_ACCOUNT_JSON = "cloud_2_key.json"

# Initialize the Google Cloud Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient.from_service_account_file(SERVICE_ACCOUNT_JSON)

def text_to_speech(text, voice_name="en-US-Wavenet-D", speaking_rate=1.0, pitch=0.0):
    # Prepare the text input for synthesis
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure the voice parameters (change voice_name to other models like Journey Voice)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",  # You can change this to support other languages
        name=voice_name,        # This is where you choose WaveNet or other models
    )

    # Configure the audio output settings with speaking rate and pitch
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,  # Output as MP3
        speaking_rate=speaking_rate,  # Control the speed of the speech
        pitch=pitch                   # Control the pitch of the speech
    )

    # Generate speech from text
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content

def transcribe_audio(audio_file, sample_rate_hertz):
    # Load Google Cloud credentials from the service account file
    client = speech.SpeechClient.from_service_account_file(SERVICE_ACCOUNT_JSON)

    # Load audio file
    with io.open(audio_file, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code="en-US",
    )

    # Transcribe the audio
    try:
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + "\n"
        return transcript
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def extract_audio_from_video(video_file, output_audio_path):
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(output_audio_path)

def convert_stereo_to_mono(audio_file, output_audio_path):
    # Use pydub to load the audio and convert it to mono
    audio = AudioSegment.from_wav(audio_file)
    mono_audio = audio.set_channels(1)  # Convert to mono
    mono_audio.export(output_audio_path, format="wav")

def get_audio_sample_rate(audio_file):
    # Use pydub to get the sample rate of the audio file
    audio = AudioSegment.from_wav(audio_file)
    return audio.frame_rate

def replace_audio_in_video(video_file, new_audio_file, output_video_file):
    # Load the original video
    video = mp.VideoFileClip(video_file)
    
    # Load the new audio
    new_audio = mp.AudioFileClip(new_audio_file)
    
    # Set the new audio to the video, remove the original audio
    final_video = video.set_audio(new_audio)
    
    # Write the final video to the output path
    final_video.write_videofile(output_video_file, codec="libx264", audio_codec="aac")

def get_video_duration(video_file):
    video = mp.VideoFileClip(video_file)
    return video.duration * 1000  # Convert to milliseconds

def get_audio_duration(audio_file):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file)
    # Return the duration in milliseconds
    return len(audio)

# Streamlit UI
st.title("Video to Text Transcription")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    # Save the uploaded video temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Extracting audio from video...")
    audio_path = "temp_audio.wav"
    extract_audio_from_video(temp_video_path, audio_path)

    st.write("Converting stereo audio to mono...")
    mono_audio_path = "mono_audio.wav"
    convert_stereo_to_mono(audio_path, mono_audio_path)

    st.write("Getting audio sample rate...")
    sample_rate = get_audio_sample_rate(mono_audio_path)

    st.write(f"Detected sample rate: {sample_rate} Hz")

    st.write("Transcribing audio...")
    transcript = transcribe_audio(mono_audio_path, sample_rate)

    if transcript:
        azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"  # Replace with your actual key
        azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
        
        headers = {
            "Content-Type": "application/json",  # Specifies that we are sending JSON data
            "api-key": azure_openai_key  # The API key for authentication
        }
        
        def response_api(query):
            # data = {
            #     "messages": [
            #         {
            #             "role": "system",
            #             "content": "You are an AI assistant who will rewrite the same user input with corrected grammatical and spelling mistakes."
            #         },
            #         {
            #             "role": "user",
            #             "content": query
            #         }
            #     ],
            #     "max_tokens": 2000
            # }
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

            system = """You are an AI assistant who will rewrite the same user input with corrected grammatical and spelling mistakes. """
            grade_prompt = ChatPromptTemplate.from_messages(
                  [
                      ("system", system),
                      ("human", "User question: {question}"),
                  ]
              )
            grader = grade_prompt | llm
            response = grader.invoke({"question":query})

            return response.content

        transcript_correct = response_api(transcript)

        if transcript_correct:
            st.write("Transcript:")
            st.text_area("Transcript", transcript_correct)

            # Generate new audio from transcript
            audio_content = text_to_speech(transcript_correct, voice_name="en-US-Wavenet-D",speaking_rate=0.65, pitch=1.0)
            new_audio_path = "new_audio.mp3"
            with open(new_audio_path, "wb") as out_f:
                out_f.write(audio_content)
            
            st.audio(io.BytesIO(audio_content), format="audio/mp3")
            st.download_button(label="Download Audio", data=audio_content, file_name="speech.mp3", mime="audio/mp3")

        #     # Replace the original audio in the video with the new one
        #     output_video_path = "output_video.mp4"
        #     st.write("Replacing original audio with new audio in the video...")
        #     replace_audio_in_video(temp_video_path, new_audio_path, output_video_path)

        #     st.video(output_video_path)
        #     st.download_button(label="Download Video", data=open(output_video_path, "rb"), file_name="output_video.mp4", mime="video/mp4")

        # # Cleanup files only if they exist
        # for path in [temp_video_path, audio_path, mono_audio_path, new_audio_path]:
        #     if os.path.exists(path):
        #         os.remove(path)
    else:
        st.error("Failed to transcribe the audio.")
