module.exports = {
  run: [
    // 1. Update the launcher scripts (this repo)
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // 2. Update the upstream SongGeneration app
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
  ]
}
