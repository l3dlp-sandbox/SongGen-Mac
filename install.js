module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone the base repository into 'app'
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/tencent-ailab/SongGeneration app"
        ]
      }
    },
    // 2. Download Model Weights (11GB)
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-Runtime"],
        "local-dir": "."
      }
    },
    // 3. Cleanup Cache to save disk space
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "rm -rf cache",
          "rm -rf .cache"
        ]
      }
    },
    // 4. Install Dependencies (Mac Optimized)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install torch torchaudio torchvision",
          "uv pip install -r ../requirements_mac.txt"
        ]
      }
    },
    // 5. APPLY FIXES: Copy your local patched files into the app folder
    { method: "fs.copy", params: { src: "main.py", dest: "app/main.py" } },
    { method: "fs.copy", params: { src: "config.py", dest: "app/config.py" } },
    { method: "fs.copy", params: { src: "schemas.py", dest: "app/schemas.py" } },
    { method: "fs.copy", params: { src: "generation.py", dest: "app/generation.py" } },
    { method: "fs.copy", params: { src: "model_server.py", dest: "app/model_server.py" } },
    
    // --- CRITICAL FIX: Use the correct inference engine ---
    { method: "fs.copy", params: { src: "levo_inference.py", dest: "app/tools/gradio/levo_inference.py" } },

    // 6. UPDATE UI: Copy the Duration Slider components
    { method: "fs.copy", params: { src: "app.js", dest: "app/web/static/app.js" } },
    { method: "fs.copy", params: { src: "components.js", dest: "app/web/static/components.js" } },
    
    // 7. Finish
    {
      method: "notify",
      params: {
        html: "Installation Complete! Click 'Start' to run SongGen-Mac."
      }
    }
  ]
}