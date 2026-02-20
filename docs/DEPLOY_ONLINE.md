# NEXUS Online Deployment (GitHub + Render)

## 1) Push code to GitHub

```bash
git add .
git commit -m "feat: bilingual dashboard + realtime logs + render deploy config"
git push origin main
```

## 2) Deploy full app on Render

1. Open `https://dashboard.render.com/`.
2. Click `New` -> `Blueprint`.
3. Connect your GitHub repo and select this repository.
4. Render will detect `render.yaml`.
5. Set secret env var:
   - `ZAI_API_KEY`
6. Deploy.

After deploy, Render gives a public URL like:

`https://nexus-quant-dashboard.onrender.com`

## 3) Verify

Open:

- `/` -> Dashboard UI
- `/api/status` -> Health JSON
- `/api/stream` -> SSE live updates
- `/api/log_stream?target=dashboard` -> realtime log stream

## Notes

- `GitHub Pages` is static hosting only and cannot run FastAPI backend endpoints.
- For full realtime dashboard + AI chat + logs, use Render (or Railway/Fly.io).
