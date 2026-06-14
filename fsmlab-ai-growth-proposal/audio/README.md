# 真人感語音音檔

這份 HTML PPT 已依 `../audio-src/slide-XX.txt` 產出逐頁真人感 neural voice 音檔。

目前音檔：

```text
audio/slide-01.m4a
audio/slide-02.m4a
...
audio/slide-13.m4a
```

來源規格：

- Voice: `zh-TW-HsiaoChenNeural`
- Format: M4A / AAC / 44.1 kHz / mono / 128 kbps
- 原則：不用 macOS `say` 或機器感預設語音。

若後續 Jason 提供真人錄音或指定授權真人音源，可用同名檔案覆蓋。若需要從一段已核准真人音源切出片段，可使用：

```bash
ffmpeg -y \
  -ss 00:00:03.200 \
  -to 00:00:12.800 \
  -i approved-human-voice-source.m4a \
  -vn -ac 1 -ar 44100 -c:a aac -b:a 128k \
  audio/slide-01.m4a
```
