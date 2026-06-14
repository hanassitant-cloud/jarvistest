# 真人語音音檔待補

這份 HTML PPT 已完成逐頁真人朗讀稿與播放器結構，但目前尚未放入真人錄音檔。

請依 `../audio-src/slide-XX.txt` 錄製或取得 Jason 核准的真人語音後，輸出為：

```text
audio/slide-01.m4a
audio/slide-02.m4a
...
audio/slide-13.m4a
```

不要用機器感 TTS 冒充真人語音。若需要從一段已核准真人音源切出片段，可使用：

```bash
ffmpeg -y \
  -ss 00:00:03.200 \
  -to 00:00:12.800 \
  -i approved-human-voice-source.m4a \
  -vn -ac 1 -ar 44100 -c:a aac -b:a 128k \
  audio/slide-01.m4a
```

