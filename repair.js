module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "echo '--- STARTING REPAIR ---'",
          
          // 1. Repair the Web Folder (Fixes 'Babel is not defined')
          "cp -R web app/",
          
          // 2. Force update the UI scripts
          "cp -f app.js app/web/static/",
          "cp -f components.js app/web/static/",
          
          // 3. Force update the Backend scripts
          "cp -f main.py app/main.py",
          "cp -f config.py app/config.py",
          "cp -f schemas.py app/schemas.py",
          "cp -f generation.py app/generation.py",
          "cp -f model_server.py app/model_server.py",
          
          // 4. Force update the Inference Engine
          "cp -f levo_inference.py app/tools/gradio/levo_inference.py",
          
          "echo '--- REPAIR COMPLETE ---'"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Repair Complete! Refresh your browser and try Starting again."
      }
    }
  ]
}