# Deploy — ChGK Model website

Production deployment to a single VPS, behind nginx, in two containers.

```
       internet
          │
          ▼
┌─────────────────────┐
│ docker bridge net   │
│                     │
│  nginx :80   ───►   │ app (uvicorn) :8000
│  (chgk-nginx)       │ (chgk-app)
│                     │   ▲
└─────────────────────┘   │ bind-mount, ro
                          │
                /srv/chgk-model/data/chgk.duckdb
```

- **Image** is built locally (`linux/amd64`), shipped via `docker save | ssh docker load`. No registry needed.
- **DuckDB file** lives on the host (`/srv/chgk-model/data/chgk.duckdb`) and is bind-mounted read-only — so model rebuilds only need to re-rsync the file.
- **Admin** (`/admin/reload-db`) is protected by `ADMIN_TOKEN` from `.env`.
- **TLS** isn't configured yet (HTTP only, IP). See *Adding HTTPS* below.

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | App + nginx services, bridge network |
| `nginx.conf` | Reverse-proxy vhost (gzip, sane proxy headers) |
| `.env.example` | Template for the secrets file (admin token) — lives on the **server** as `.env` |
| `.deploy.env.example` | Template for the local deploy script settings (host, user, paths) — copy to `.deploy.env` (gitignored) on the **workstation** |
| `deploy.sh` | Local script: build image → ship → restart stack |
| `refresh-db.sh` | Local script: rsync DB → atomic swap → restart `app` |

## One-time workstation setup

```bash
cd website/deploy
cp .deploy.env.example .deploy.env
$EDITOR .deploy.env   # set REMOTE_HOST=… (your server IP/hostname)
```

`.deploy.env` is gitignored, so the server address never lands in the repo.
Anything you `export` in your shell (e.g. `REMOTE_HOST=staging.example.org`)
overrides what's in the file for one-off deploys.

## One-time server prep

```bash
ssh -i ~/.ssh/id_rsa root@<your-server>

# Docker (skip if already installed).
curl -fsSL https://get.docker.com | sh

# Layout.
mkdir -p /srv/chgk-model/data
```

## First deploy (from the workstation)

```bash
cd /Users/fbr/Projects/personal/сhgk-model

# 1. Build the DB locally (you already do this).
python -m website.build.build_db \
  --cache data.npz --results results/seq.npz \
  --out website/data/chgk.duckdb

# 2. Push everything (first run uploads the .env template too — edit it).
./website/deploy/deploy.sh --db
# → on the FIRST run the script will copy .env.example to the server as
#   .env and exit, asking you to set ADMIN_TOKEN.  Edit it:
ssh root@<your-server> \
  "python3 -c 'import secrets;print(\"ADMIN_TOKEN=\"+secrets.token_urlsafe(32))' > /srv/chgk-model/.env && chmod 600 /srv/chgk-model/.env && cat /srv/chgk-model/.env"

# 3. Re-run the deploy with --db to upload the DB and start everything.
./website/deploy/deploy.sh --db
```

After this, `http://<your-server>/` should serve the site.

## Routine updates

| Change                                | Command                                  |
|---------------------------------------|------------------------------------------|
| Just the DB (after model retrain)     | `./website/deploy/refresh-db.sh`         |
| App code only (templates, routes)     | `./website/deploy/deploy.sh --image-only`|
| Full redeploy (code + compose files)  | `./website/deploy/deploy.sh`             |
| Code + DB                             | `./website/deploy/deploy.sh --db`        |

`--image-only` skips uploading `docker-compose.yml`/`nginx.conf` and only restarts the `app` service — the safest update if you're just iterating on Python or templates.

## Useful remote commands

```bash
ssh -i ~/.ssh/id_rsa root@<your-server>

cd /srv/chgk-model
docker compose ps
docker compose logs -f --tail=200 app
docker compose logs -f --tail=200 nginx

# Hot-reload DB without restart (single-worker app, so this works):
curl -X POST -H "X-Admin-Token: $(grep ADMIN_TOKEN .env | cut -d= -f2)" \
  http://127.0.0.1/admin/reload-db

# Inspect DB on the server.
docker compose exec app python -c \
  "import duckdb; print(duckdb.connect('/app/data/chgk.duckdb', read_only=True).execute('SELECT COUNT(*) FROM players').fetchall())"
```

## Adding HTTPS later

Two easy options:

1. **Inside the same compose** — add a `certbot` container + a `letsencrypt` named volume, and either swap `nginx.conf` for the standard `webroot` ACME challenge layout, or use `nginx-proxy/acme-companion`.
2. **Cloudflare in front** — point the domain at Cloudflare with `Proxy=on` and Origin Certificate.  Server stays on plain HTTP behind Cloudflare's edge.

Option 2 is the lowest-effort one once a domain is registered.

## What about hardening?

The nginx vhost intentionally proxies `/admin/*` too — those endpoints are token-protected by the app and the token is a 32-byte secret in `.env`.  No nginx-level allow/deny because the docker bridge eats the real client IP.

If you want extra paranoia, change `docker-compose.yml` to publish the app on `127.0.0.1:8000` (instead of leaving it only on the docker bridge) and let nginx talk to `host.docker.internal:8000` — then add `allow 127.0.0.1; deny all;` on `/admin/`. Not bothering by default.

## Troubleshooting

- **`deploy.sh` errors at "ADMIN_TOKEN must be set"** — edit `/srv/chgk-model/.env` on the server, then re-run.
- **`deploy.sh` errors at the buildx step** — make sure Docker Desktop is running with buildx enabled (it is by default since Docker 23).
- **Server returns 502** — the app container probably crashed. `docker compose logs --tail=200 app` will show the traceback. Most common causes: missing DB at `/app/data/chgk.duckdb`, or schema mismatch between the rebuilt DB and the templates.
- **`refresh-db.sh` finishes but the page still shows old data** — the script restarts the `app` container; if you skipped that step manually, hot-reload via `/admin/reload-db` is the alternative. With multiple workers only one would reload — that's why we run with `--workers 1`.
