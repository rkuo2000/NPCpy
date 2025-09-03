import os
def tts_elevenlabs(text, 
                   api_key=None,
                   voice_id='JBFqnCBsd6RMkjVDRZzb', 
                   model_id='eleven_multilingual_v2',
                   output_format= 'mp3_44100_128'):
    if api_key is None:
        api_key = os.environ.get('ELEVENLABS_API_KEY')
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play

    client = ElevenLabs(
        api_key=api_key,
    )

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format= output_format
    )

    play(audio)
    return audio
