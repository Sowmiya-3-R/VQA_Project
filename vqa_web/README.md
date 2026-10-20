# VQA web demo (VisionQA page + trained baseline)

Layout inside your repo (copy the `app/` folder into the repo root):

    VQA_Project/
      app/
        server.py          FastAPI: serves the page and runs the model (POST /api/predict)
        static/            the web page (index.html, styles.css, script.js)
      models/  text_encoder/  ...   (already in the repo)

Files you need next to it (do NOT commit them): best_model.pt, vocab.json, answers.json, projection.pt

Install (once):
    pip install fastapi uvicorn python-multipart torch torchvision pillow

Run from the repo root:
    python app/server.py --checkpoint best_model.pt --vocab vocab.json \
        --answers answers.json --projection projection.pt
Open http://127.0.0.1:8000

Before the first run:
- projection.pt comes from recover_projection.py (it recovers the random 2048->1024 layer used for the training features).
- Check the tokenizer function name: grep -n "def " text_encoder/tokenizer.py
  server.py imports encode_question; change that line if your function has another name.
- Check the validation error printed by recover_projection.py (about 1e-4 or lower is good).
