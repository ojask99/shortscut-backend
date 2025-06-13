from pytubefix import YouTube
from pytubefix.cli import on_progress

url1 = "https://youtu.be/B2ks-M9RoW4?si=56ZpSA4zcYfV4hnw"
url2 = "https://youtu.be/LMoi4A2RexE?si=VpXLz867hUN3VDR7"

yt = YouTube(
    url2,
    on_progress_callback=on_progress 
)

print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()

print("Download completed.")



