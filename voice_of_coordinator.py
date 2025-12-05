import os
from dotenv import load_dotenv
from gtts import gTTS
import subprocess
import platform

# load env
load_dotenv()
ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY")

# Step1b: Setup Text to Speech–TTS–model with ElevenLabs (optional)
try:
    import elevenlabs
    from elevenlabs.client import ElevenLabs
except Exception:
    # If elevenlabs package missing, program will still work using gTTS fallback
    elevenlabs = None
    ElevenLabs = None

def text_to_speech_with_gtts_old(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)

def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    if ElevenLabs is None:
        raise RuntimeError("ElevenLabs library not available")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

# Step2: Use Model for Text output to Voice

def _autoplay_file(output_filepath):
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            # aplay works only for WAV on some systems; you may use mpg123 or ffplay if needed
            subprocess.run(['aplay', output_filepath])
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

def text_to_speech_with_gtts(input_text, output_filepath):
    language = "en"
    audioobj = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    _autoplay_file(output_filepath)

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    if ElevenLabs is None:
        raise RuntimeError("ElevenLabs library not available")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    _autoplay_file(output_filepath)

# Example usage (commented out by default)
# input_text="Hi this is Ai with proma!"
# text_to_speech_with_gtts_old(input_text=input_text, output_filepath="gtts_testing.mp3")
# text_to_speech_with_elevenlabs_old(input_text, output_filepath="elevenlabs_testing.mp3")
