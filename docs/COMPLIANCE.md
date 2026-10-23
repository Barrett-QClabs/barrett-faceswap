[English](COMPLIANCE.md) | [中文](COMPLIANCE.zh-CN.md)

# Compliance

This is not legal advice. It is the behavior of the code in `src/barrett_faceswap/compliance/`, plus the limits operators have to respect on their own.

## Master switch

`compliance.enabled` defaults to **false**.

While it is false, `CompliancePolicy.before_process` and `after_process` still run, and they skip the consent requirement, the watermark, content-filter blocks, and the audit log. They log one warning: enforcement is off, and a real likeness needs it on. `--enable-compliance` turns enforcement on for that invocation even if the YAML says false. There is no flag that turns a `true` config back off.

The nested block in `configs/default.yaml` is the profile that becomes active when you flip the switch:

- a consent record is mandatory
- a visible label and embedded metadata are written
- the content filter runs (default provider `stub`)
- every completed, refused, or failed run appends one JSON line

Unit tests construct a config with `enabled: true` and exercise those paths. A dry-run of the shipped file does not.

## Acceptable use

- A source face is used only with that person's authorization for this use.
- No non-consensual face swaps.
- No sexual deepfakes.
- No sexual content involving minors, and no face swaps of minors.
- No fraud, impersonation, or other deceptive use.
- Synthetic output that you share is labeled as synthetic.

The filter and the watermark do not replace the consent record.

## Consent

`compliance/consent.py` accepts either:

- `--consent-file` pointing at a JSON object with `subject_id`, `statement` (at least 10 characters), `consented_on` (`YYYY-MM-DD`), and `scope`. Optional `expires_on`. An expired record is refused.
- `--acknowledge-authorized-use` together with `--consent-subject`. The acknowledgement text defaults to a catalog sentence when you omit `--consent-statement`.

The audit stores a `consent_id` (`file:` or `ack:` plus a short hash). Passing both forms at once is refused. `examples/consent.example.json` is a template, not an authorization.

## Watermark

`compliance/watermark.py` draws a bottom bar with the active label. The default label comes from `watermark.label` in the catalog: `AI-generated` in English and `人工智能生成` in Chinese. Set `compliance.watermark.text` to override it.

PNG output also gets a `tEXt` chunk (`barrett.faceswap.ai_generated=true` and a Description). JPEG output gets an EXIF ImageDescription. A `.meta.json` sidecar is written next to the file.

A video run watermarks every output frame when enforcement and the watermark are on, then calls `apply_video_metadata` to set the container comment and write the sidecar. Filter checks use a sample (`video.filter_every_n_frames`, plus the last frame), not every frame. If the metadata ffmpeg call fails, the run still succeeds: the pixels are already marked and the sidecar is written. If ffmpeg is missing entirely, frame extraction fails before encode and the sidecar from a still-image run is still written. The visible bar uses the Latin catalog string. OpenCV's built-in font cannot draw `人工智能生成`. The Unicode label is what the container comment and the sidecar store.

`--no-watermark` is the explicit, logged way to skip the label while enforcement is on. A config that sets `watermark.enabled: false` without that flag is refused, so a YAML edit alone cannot drop the label quietly.

## Content filter

`compliance/content_filter.py` defines `ContentFilter.assess` and a registry of builders.

| `provider` | Behavior |
| --- | --- |
| `stub` | Default. Allows the frame and logs that no real detector is loaded. |
| `none` | Filter off. This is a weakening. |
| `nudenet` | Imported only when selected. Not a project dependency. |
| `custom` | Loads `module:factory` from `content_filter.custom_entrypoint`. |

`content_filter.threshold` and `content_filter.block_on_minor` ship as `0.6` and `true`.

The factory receives a `FilterSettings` object and returns something with `assess(image, role=...) -> FilterDecision`. A decision can set `allowed`, `nsfw_flag`, and `minor_flag`. `supports_minor_detection` is false for `stub` and for NudeNet. NudeNet does not classify age. A custom provider that sets `minor_flag` is blocked when `block_on_minor` is true. The stub does not pretend it saw a minor; it says minor detection is unavailable.

### Turning a real detector on

Edit the YAML (this is not a CLI override, so it does not need the permission variable):

```yaml
compliance:
  enabled: true
  content_filter:
    enabled: true
    provider: nudenet
    threshold: 0.6
    block_on_minor: true
```

Then `pip install nudenet`. The package is opt-in. If it is missing, the run stops with an install hint. NudeNet still does not satisfy a minor check. Treat that as an operator duty, not as a green light.

A custom module:

```python
def build(settings):
    class Gate:
        name = "custom"
        supports_minor_detection = True
        def assess(self, image, *, role):
            ...
    return Gate()
```

```yaml
content_filter:
  provider: custom
  custom_entrypoint: "my_package.filters:build"
```

### Permission switch

Disabling the filter, setting the provider to `none`, raising the threshold, turning `block_on_minor` off, or changing the provider from the CLI (`--content-filter-provider`, `--disable-content-filter`, `--no-block-on-minor`) requires one of:

- environment variable `BARRETT_FACESWAP_ALLOW_FILTER_DISABLE=1`
- `configs/admin.toml` containing `allow_filter_disable = true` (this file is gitignored)

Without that switch the CLI refuses and the attempt is audited when enforcement is on. The same switch is required when the YAML itself sets `enabled: false`, `provider: none`, or `block_on_minor: false` under the content filter. Putting `provider: nudenet` in the YAML is the documented opt-in and does not use this variable.

Every permitted weakening stores `permission_used` and the reason codes in the audit line.

The filter is defense in depth. It does not authorize a face, and the stub provider does not review one.

## Audit log

`compliance/audit_log.py` appends one JSON object per line to `compliance.audit.path` (default `outputs/audit/audit.jsonl`). The writer opens the file in append mode and locks it on Unix. It does not rewrite earlier lines.

Fields include the timestamp, event (`completed`, `refused`, `failed`), input hashes, consent id, output path, watermark flags, and the filter decision. Enforcement-off runs do not write this file. If the log cannot be appended during an enforced run, the run fails.

## Legal notes

Operators have to follow the law where they are. The notes below are pointers, not an opinion on any particular deployment.

**China.** 《互联网信息服务深度合成管理规定》(Provisions on the Administration of Deep Synthesis of Internet-based Information Services, effective 10 January 2023) requires providers of deep-synthesis services to label synthetic content, especially where the result could be mistaken for a real recording, and it prohibits using deep synthesis for illegal purposes. 《生成式人工智能服务管理暂行办法》 also expects generative-AI services to mark generated content. The watermark path is this project's technical answer to a labeling duty. Whether a given deployment is "providing a service" under those rules is a question for counsel.

**European Union.** The AI Act's transparency rules (Article 50) expect deployers of systems that generate or manipulate image, audio, or video constituting a deep fake to disclose that the content was artificially generated or manipulated. Artistic and similar works have a narrower disclosure path. Again, this is a pointer to the instrument, not a compliance conclusion.

**Weights.** InsightFace's library code is MIT. Its pretrained models, including `buffalo_l` and `inswapper_128.onnx`, are for non-commercial research unless InsightFace grants otherwise. Their 2025-11-24 update says to contact `contact@insightface.ai` for the inswapper series and `recognition-oss-pack@insightface.ai` for the open recognition packs. SimSwap is CC BY-NC 4.0 and is not shipped here. See [RESEARCH.md](RESEARCH.md).

## What the code will not do for you

It will not decide that a template JSON file is real consent. It will not detect minors when the provider is `stub` or NudeNet. It will not label output while the master switch is off. Turn the switch on before a real face is processed.
