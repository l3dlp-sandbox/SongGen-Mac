module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
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
    // 6. Finish
    {
      method: "notify",
      params: {
        html: "Installation Complete! Click 'Start' to run SongGen-Mac."
      }
    }
  ]
}
