# Image Runtime

This skill can use different image backends. Keep the deck workflow stable and only swap the renderer.

## Token Safety

Never hardcode or save API tokens. Use environment variables:

- `GIIISP_AUTH_TOKEN` for Giiisp/SiTian Imagine (apply or refresh at `https://giiisp.com/#/mcp/authenticate`).
- Project-specific env vars only when working inside a repo that already defines them.

Manifests may record `token_used: true`, but never the token value.

## Codex Imagegen

When the `image_gen` tool is available, use it directly for visual-target generation and extraction-based layer generation.

Rules:

- First generate or receive `visual_targets/NN.png`: a finished, high-quality slide image with complete intended composition and page text.
- Before extracting background, frame, or icons from a target, show the target image with `view_image` so the image model can treat it as the current edit target.
- Generate background, frame, and icon/decor assets as separate raster layers from the visual target.
- Keep ordinary editable text out of extracted background and frame layers.
- Save or copy generated layer assets into `layers/NN/`.
- Record source path or tool output identifier in `layer_manifest.json`.

## Giiisp/SiTian Imagine

Use when the user provides a Giiisp/SiTian token or `GIIISP_AUTH_TOKEN` is already set.

Endpoint summary:

- Generate: `POST http://images.sitianai.com/api/generate-async`
- Poll: `GET http://images.sitianai.com/api/generate-jobs/{job_id}`
- Header: `Authorization: Bearer <token>`

Common request fields:

```json
{
  "prompt": "full prompt",
  "negativePrompt": "watermark, blurry text, placeholder text",
  "aspectRatio": "16:9",
  "imageSize": "1K",
  "numberOfImages": 1,
  "responseModalities": ["IMAGE", "TEXT"],
  "outputMimeType": "image/png"
}
```

Reference/edit images may use:

- `referenceImages`
- `imageBase64`
- `imageMimeType`

If the API returns token errors or no `job_id`, write a blocker in the run directory and do not fake output images. When the token is missing, invalid, or expired, tell the user to authenticate at `https://giiisp.com/#/mcp/authenticate`. When `DASHSCOPE_API_KEY` is missing for VLM review or content generation, tell the user to apply for a DashScope/Bailian API key at `https://help.aliyun.com/zh/model-studio/get-api-key`.

For PPT deck generation, request `aspectRatio: "16:9"` and default to `imageSize: "1K"` unless the user explicitly asks for high-resolution output. Treat `2K` as an optional final regeneration/upscale pass after the slide has passed VLM review. Before packaging a PPT deck, inspect the returned image dimensions. Some backends may accept `aspectRatio: "16:9"` but still return a square image such as `1024 x 1024`. That image can be valid for standalone figure generation but is invalid for a 16:9 slide image. Do not stretch it into PPTX; record the aspect-ratio failure and regenerate with a backend/configuration that returns a native wide image, or stop for user choice.

## Project Providers

When working inside an app that already has image provider code, prefer its provider wrapper over adding a second HTTP client.

Record:

- provider name
- model or route
- prompt file
- output layer image path
- failures and alternate backend path

## Prompt Guidance

Per-slide prompts should state:

- exact slide text
- composition and layout
- aspect ratio
- language
- style brief
- what must not appear

For `visual_target` prompts, include the intended slide text so the model can design the full page. For background/frame/icon extraction prompts, remove ordinary text from image layers and put ordinary wording into PPT text boxes.
