import subprocess

def reformat_audio(audio_file):
    result = subprocess.run(
        ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', 'pipe:1'],
        input=audio_file,
        capture_output=True
    )
    return result.stdout