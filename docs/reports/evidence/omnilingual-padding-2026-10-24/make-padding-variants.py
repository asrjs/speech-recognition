import os
import wave

root = os.getcwd()
source = os.path.join(root, "tests", "fixtures", "jfk2.en.wav")
out_dir = os.path.join(root, "tmp", "omnilingual-diagnosis")
with wave.open(source, "rb") as inp:
    params = inp.getparams()
    audio = inp.readframes(inp.getnframes())
    silence = bytes(params.sampwidth * params.nchannels * params.framerate * 5)
for name, pcm in (("left5", silence + audio), ("right5", audio + silence), ("both5", silence + audio + silence)):
    path = os.path.join(out_dir, f"jfk2-{name}.wav")
    with wave.open(path, "wb") as out:
        out.setparams(params)
        out.writeframes(pcm)
    print(path, os.path.getsize(path))

one_second_silence = bytes(params.sampwidth * params.nchannels * params.framerate)
for name, pcm in (("left1", one_second_silence + audio), ("right1", audio + one_second_silence)):
    path = os.path.join(out_dir, f"jfk2-{name}.wav")
    with wave.open(path, "wb") as out:
        out.setparams(params)
        out.writeframes(pcm)
    print(path, os.path.getsize(path))
