import asyncio
import edge_tts

def generate_voice(text, audio_path, voice="en-IN-NeerjaNeural"):

    async def _save():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_path)

    asyncio.run(_save())
    return audio_path
