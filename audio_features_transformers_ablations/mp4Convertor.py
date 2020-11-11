from moviepy.editor import *
from moviepy.config import change_settings
change_settings({"FFMPEG_BINARY": "ffmpeg"})

def convertVid(mp4, mp3):
    videoclip = VideoFileClip(mp4)
    audioclip = videoclip.audio
    audioclip.write_audiofile(mp3, codec='libmp3lame')
    audioclip.close()
    videoclip.close()
mp4 = "Desktop/v_CricketBowling_g20_c05.mp4"
mp3 = "Desktop/cricket_bowl_test.mp3"
convertVid(mp4, mp3)