module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Download Model Weights (~15GB)
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-Runtime"],
        "local-dir": "."
      }
    },
    // 2. Cleanup Cache to save space
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
    // 3. Install Dependencies
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
    // 4. Finish
    {
      method: "notify",
      params: {
        html: "Installation Complete! Click 'Start' to run SongGen-Mac."
      }
    }
  ]
}
