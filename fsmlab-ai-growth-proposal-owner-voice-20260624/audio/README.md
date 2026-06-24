# 老闆真人影片原聲音檔

這份 HTML PPT 是新建的 `owner-voice` 版本，未覆蓋原本的飛創美簡報。

音檔來源為 Jason 於 2026-06-24 提供的飛創美老闆直式訪談影片，已抽出原聲、音量標準化，並切成逐頁可播放片段。

目前音檔：

```text
audio/owner-full.m4a
audio/slide-01.m4a
audio/slide-02.m4a
...
audio/slide-13.m4a
```

來源規格：

- Voice: FSMLAB 老闆影片原聲
- Source video: `../assets/video/fsmlab-owner-source.mp4`
- Format: M4A / AAC / mono / 128 kbps
- 原則：本版保留影片真人聲音，不使用 macOS `say` 或機器感預設語音。

若後續 Jason 提供逐頁正式錄音，可用同名檔案覆蓋。若需要從一段已核准真人音源切出片段，可使用：

```bash
ffmpeg -y \
  -ss 00:00:03.200 \
  -to 00:00:12.800 \
  -i approved-human-voice-source.m4a \
  -vn -ac 1 -ar 44100 -c:a aac -b:a 128k \
  audio/slide-01.m4a
```
