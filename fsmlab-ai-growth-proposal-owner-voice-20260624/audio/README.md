# 粉衣老闆聲線 AI 語音導覽音檔

這份 HTML PPT 是新建的 `owner-voice` 版本，未覆蓋原本的飛創美簡報。

音檔使用 Jason 於 2026-06-24 提供的飛創美老闆直式訪談影片作為授權聲音樣本，僅取粉色衣服老闆段落建立參考音，並用 F5-TTS 依 `../audio-src/slide-XX.txt` 產生逐頁 AI 語音導覽。

目前音檔：

```text
audio/owner-full.m4a
audio/slide-01.m4a
audio/slide-02.m4a
...
audio/slide-13.m4a
```

來源規格：

- Voice reference: FSMLAB 粉色衣服老闆影片聲音樣本
- Source video: `../assets/video/fsmlab-owner-source.mp4`
- Generator: local F5-TTS voice clone workflow
- Format: M4A / AAC / mono / 128 kbps
- 原則：本版是授權聲音樣本 AI 語音導覽，不是影片原聲切片，不宣稱真人現場錄音；不使用影片後段白色衣服講者，也不使用 macOS `say` 或機器感預設語音。

若後續 Jason 提供逐頁正式錄音，可用同名檔案覆蓋。若需要從一段已核准真人音源切出片段，可使用：

```bash
ffmpeg -y \
  -ss 00:00:03.200 \
  -to 00:00:12.800 \
  -i approved-human-voice-source.m4a \
  -vn -ac 1 -ar 44100 -c:a aac -b:a 128k \
  audio/slide-01.m4a
```
