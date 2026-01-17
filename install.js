module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone the base repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/tencent-ailab/SongGeneration app"
        ]
      }
    },
    // 2. Download model weights (11GB)
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-Runtime"],
        "local-dir": "."
      }
    },
    // 3. Cleanup Cache to save space
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
    // 4. Install Dependencies
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
    // 5. FORCE COPY FILES
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "echo '--- STARTING FILE SYNC ---'",
          
          // A. Copy the entire 'web' folder structure first
          "cp -R web app/",
          
          // B. Force copy the specific UI files (overwriting the base ones)
          "cp -f app.js app/web/static/",
          "cp -f components.js app/web/static/",
          
          // C. Force copy all Python Backend files
          "cp -f main.py app/main.py",
          "cp -f config.py app/config.py",
          "cp -f schemas.py app/schemas.py",
          "cp -f generation.py app/generation.py",
          "cp -f model_server.py app/model_server.py",
          "cp -f models.py app/models.py",
          "cp -f gpu.py app/gpu.py",
          "cp -f sse.py app/sse.py",
          "cp -f timing.py app/timing.py",
          
          // D. Force copy the Inference Engine (Critical for Mac)
          "cp -f levo_inference.py app/tools/gradio/levo_inference.py",
          
          "echo '--- FILE SYNC COMPLETE ---'"
        ]
      }
    },
    // 6. Finish
    {
      method: "notify",
      params: {
        html: "Installation Complete! Click 'Start' to run SongGen-Mac."
      }
    }
  ]
}
