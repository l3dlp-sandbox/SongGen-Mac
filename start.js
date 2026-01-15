module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          "PYTORCH_ENABLE_MPS_FALLBACK": "1",
          "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"
        },
        message: [
          "python main.py --host 127.0.0.1 --port {{port}}"
        ],
        // FIX: 'on' must be an Array [ { ... } ]
        on: [{
          event: "/(http:\/\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        "url": "{{input.event[0]}}"
      }
    }
  ]
}