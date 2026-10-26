"""Apply the packaged FLUX 3 workflow to existing quick-edit requests."""
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image


class QuickEditWorkflow:
    def __init__(self, workflow_directory, input_directory, default_size):
        root = Path(workflow_directory)
        self.template = json.loads((root / "flux3_quick_edit_api.json").read_text(encoding="utf-8"))
        self.defaults = json.loads((root / "flux3_official_defaults.json").read_text(encoding="utf-8"))
        self.input_directory = Path(input_directory)
        self.default_size = default_size

    def __call__(self, request):
        graph = request.get("prompt", {})
        expected = {"76": "LoadImageWithoutListDir", "126": "LoadImageWithoutListDir",
                    "129": "LoadImageWithoutListDir", "105": "UNETLoader", "106": "CLIPLoader",
                    "107": "CLIPTextEncode", "118": "AspectRatioSizeNodeOfUtils", "94": "SaveImage"}
        if not isinstance(graph, dict) or not all(graph.get(key, {}).get("class_type") == value for key, value in expected.items()):
            return request
        model = graph["105"]["inputs"].get("unet_name")
        if model not in ("flux-2-klein-4b-fp8.safetensors", "flux3-image-klein-int8-convrot.safetensors"):
            return request
        if graph.get("102", {}).get("class_type") not in ("SamplerCustomAdvanced", "KSampler"):
            return request
        ids = {key: key if key in graph else key.replace(":", "_") for key in self.template}
        prompt = graph["107"]["inputs"]["text"]
        if isinstance(prompt, list):
            prompt = graph[prompt[0]]["inputs"]["prompt"]
        updated = deepcopy(self.template)
        for key in ("76", "126", "129", "94", "118"):
            updated[key]["inputs"] = deepcopy(graph[key]["inputs"])
        if model == "flux-2-klein-4b-fp8.safetensors":
            updated["118"]["inputs"]["size"] = self.default_size
        seed = graph["104"]["inputs"]
        updated["104"]["inputs"]["value"] = seed.get("noise_seed", seed.get("value", 0))
        references = []
        for key in ("76", "126", "129"):
            inputs = updated[key]["inputs"]
            filename = inputs.get("input_image_path") or inputs.get("image", "")
            if not inputs.get("enabled", True) or not filename:
                continue
            path = self.input_directory / filename
            if path.is_file():
                with Image.open(path) as image:
                    if image.size == (1, 1) and "A" in image.getbands() and image.getchannel("A").getextrema() == (0, 0):
                        continue
            references.append(key)
        fallback = next((updated[key]["inputs"].get("image") for key in ("76", "126", "129") if updated[key]["inputs"].get("image")), "__quick_edit_placeholder.png")
        for key in ("76", "126", "129"):
            inputs = updated[key]["inputs"]
            if not inputs.get("enabled", True) and not inputs.get("image"):
                inputs["image"] = fallback
        defaults = self.defaults["i2i" if references else "t2i"]
        updated["99"]["inputs"]["value"] = defaults["system_prompt"]
        updated["115:76"]["inputs"] = dict(defaults["text_generate"], prompt=prompt, clip=["112", 0], system_prompt=["99", 0])
        if references:
            updated["115:76"]["inputs"]["image"] = ["127", 0]
        scales = {"76": "115:78", "126": "120", "129": "131"}
        updated["127"]["inputs"] = {f"images.image{i}": [scales[key], 0] for i, key in enumerate(references)} or {"images.image0": ["115:78", 0]}
        ref_inputs = updated["133"]["inputs"]
        updated["133"]["inputs"] = {key: value for key, value in ref_inputs.items() if not key.startswith("images.")}
        updated["133"]["inputs"].update({f"images.image{i}": [key, 0] for i, key in enumerate(references)})
        updated["102"]["inputs"].update({key: value for key, value in defaults["sampler"].items() if key not in ("seed", "control_after_generate")})
        updated["100"]["inputs"]["steps"] = defaults["sampler"]["steps"]
        edge = {"1k": 1024, "2k": 2048, "4k": 4096}[updated["118"]["inputs"]["size"]]
        for key in ("109", "123", "130"):
            updated[key]["inputs"]["megapixels"] = (edge / 1024) ** 2
        updated["118"]["inputs"].update(input_width=["119", 0] if references else edge,
                                        input_height=["119", 1] if references else edge,
                                        batch_size=["119", 2] if references else 1)
        result = deepcopy(request)
        for key, node in updated.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and value[0] in ids:
                    value[0] = ids[value[0]]
            result["prompt"][ids[key]] = node
        return result
