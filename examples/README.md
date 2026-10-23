# Examples

Put only files you are allowed to process in this folder.

This repository does not ship photographs of real people. Use
`examples/make_synthetic.py` to write solid-color test images and a consent
record whose subject is the synthetic pattern itself:

```bash
python examples/make_synthetic.py
```

That creates `examples/synthetic/source.png`, `examples/synthetic/target.png`,
and `examples/synthetic/consent.json`. The pictures are generated rectangles,
not faces. A dry-run will still execute the command path. A real swap needs
ONNX weights and a photograph you have permission to use.

`examples/consent.example.json` is a template. Replace every field before you
turn compliance on for a real subject. A template is not authorization.
