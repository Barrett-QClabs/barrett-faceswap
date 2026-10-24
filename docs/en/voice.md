# Voice clone

Face swap stays the main path. Voice conversion is a second step for a **video that already has speech**.

## Source

The page opens on **Video (face + voice, recommended)**. Upload one clip of the person. The run samples frames across it, scores each face for size, sharpness, and how frontal it is, and embeds the best ones.

Averaging is used only when it earns it. The lowest-quality third of those faces is held out. If the average of the rest matches those held-out faces better than the single best face, the identity is the average. Otherwise the best face is the identity. One usable face stays that face. BlendSwap wants a source crop, so it always uses the best frame.

Speech for the clone is the same source video, with silence and noise-like stretches removed. The voice panel on the right uses that clip automatically. Upload a file there to override it. **Photo** swaps the face only. **Photo + audio** uses the photo plus the separate clip shown next to it. A target with no audio never converts voice, even when the source video has speech.

**Library** is a fourth choice. A named profile keeps the same person: more video, photo, or audio is fused into one identity and one speech clip. The image and video tab creates the profile, adds samples, removes a sample, and deletes the profile. The camera tab only selects a profile. Swap from that profile, or keep using a one-shot upload. Profiles are stored under `library/` and are not committed.

## Camera

The camera tab uses the same choices, in the same order, and adds a saved profile at the end. **Video (face + voice, recommended)** is selected when the tab opens. Only the uploads for the selected choice are shown. Profiles are managed on the image and video tab.

Voice on that tab runs while the camera is on, the microphone is sending audio, and a reference exists. The reference is an uploaded clip when one is set. Otherwise it is speech trimmed from the source video. Photo mode has no reference. Photo + audio uses the clip next to the photo. An uploaded clip wins over speech taken from the source video.

No microphone, or no reference, stays face-only. That path does not play the microphone back. Dry run does not convert. If the OpenVoice converter is not installed, the preview keeps swapping the face and the voice line says the model is missing.

`barrett-faceswap live` captures the camera only. It does not read a microphone. Use the camera tab for one-stop face and voice.

On a file, voice runs only when all of these are true:

- the voice switch is on (or you passed `--voice-reference`)
- a reference clip is set (wav, mp3, or m4a)
- the target file is a video **and** it has an audio stream

If the target has no audio, the run is face-only. The converter is not loaded. A still image, a preview of the first frame, and `--dry-run` are face-only as well. Dry-run keeps the original track when there is one.

The Gradio switch **starts off**. Choosing a target video that has audio turns it **on**. A silent video or a still turns it **off**. `voice.enabled` in `configs/default.yaml` is the auto-enable flag for that behavior. Set it to `false` if the switch should stay off until you click it. The panel sits on the right of the large preview so the source, target, and preset column stays as it is.

## Engine

The engine is **OpenVoice V2's tone-color converter** (MyShell). Code and the V2 checkpoint are MIT. It takes the target video's own speech and a short reference clip, and it changes the speaker timbre. There is no per-speaker training and no paid API.

It was chosen over the other local stacks for this machine:

- **Seed-VC** is a strong zero-shot converter, but the project is GPL-3.0, the repo is archived, and its requirements pin PyTorch 2.4. That build has no kernel for an RTX 5090 (`sm_120`).
- **CosyVoice** `inference_vc` is real voice conversion under Apache-2.0, but the install pins an older CUDA 12.1 torch plus ModelScope, wetext, and Matcha. That fights this repo's onnxruntime-gpu CUDA 13 stack.
- OpenVoice's converter is ordinary PyTorch. On an RTX 5090, install a PyTorch wheel built with **CUDA 12.8 or newer** so `sm_120` is in the wheel. `voice.device: auto` uses CUDA when that build can see the GPU, and CPU otherwise.

The face weights in this repo are still non-commercial. The OpenVoice checkpoint itself is MIT. This repository does not download it.

`voice.watermark` stays **false**. OpenVoice can embed its own mark, which needs the separate `wavmark` package. Leave it off so the compliance watermark stays the only one, and compliance stays default-off.

## Install

Do not install OpenVoice's `requirements.txt`. It pins `numpy==1.22` and `gradio==3.48`, which break this project.

From the repository root, on a machine whose PyTorch already matches the GPU:

```bash
pip install -e ".[voice]"
pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
```

RTX 5090 (Blackwell, `sm_120`), driver that supports CUDA 12.8:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[voice]"
pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
```

If the second command replaces the CUDA torch wheel, install the `cu128` wheel again. Confirm with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

You want capability `(12, 0)` on a 5090. An older wheel prints a warning that `sm_120` is not supported and then fails at runtime.

### Weights

The converter is two files. The Hugging Face repo `myshell-ai/OpenVoiceV2` is MIT. You do not need the base-speaker embeddings for this path.

```bash
pip install -U huggingface_hub
huggingface-cli download myshell-ai/OpenVoiceV2 \
  converter/config.json converter/checkpoint.pth \
  --local-dir models/openvoice_v2
```

Expected layout:

```text
models/openvoice_v2/converter/config.json
models/openvoice_v2/converter/checkpoint.pth
```

`voice.checkpoint_dir` points at `models/openvoice_v2`. A missing file raises an install hint **before** the face swap starts, and only when a conversion would actually run.

## Controls

| Setting | Meaning |
| --- | --- |
| Clone strength | `1` uses only the converted voice. `0` keeps the original track and does not load the model. Values in between mix the two after both are fitted to the picture. |
| Variation (tau) | OpenVoice's posterior noise. `0.3` is the upstream default. Higher is more random. It is not a "more clone" knob. The timbre swap itself is the reference embedding. |

Advanced stays collapsed. Progress is the face frames first (up to about 85% when voice will run), then "Converting voice", then "Muxing audio".

Length: a small duration gap (about 15%) is time-stretched so the words stay near the mouth. A larger gap is padded or trimmed. The picture is not cut to the audio. The remux copies the video bitstream and writes AAC with `+faststart`, same as the frame encoder.

## CLI

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.mp4 \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4
```

A source video turns voice on. Speech trimmed from it is the reference. `--voice-reference` replaces that clip. `--no-voice` forces face-only. A still source does not convert voice unless you pass `--voice-reference`. `--library NAME` uses a saved profile instead of `--source`. Its fused face is the identity, and its saved speech is the reference unless you pass `--voice-reference` or `--no-voice`. `--voice-strength` and `--voice-tau` override the config for that run. A target with no audio stays face-only either way.
