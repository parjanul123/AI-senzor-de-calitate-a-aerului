# Ghid Deploy Railway - Air Quality AI

## 📋 Pregătire Inițială

### 1. Verificare Dependințe
✅ `requirements.txt` - Include FastAPI și Streamlit
✅ `Dockerfile` - Creat și optimizat pentru Railway  
✅ `Dockerfile.streamlit` - Container separat pentru interfață
✅ `railway.toml` - Configurație Railway
✅ `railway.json` - Configurație alternativă Railway
✅ `.dockerignore` - Exclude fișiere inutile
✅ `/health` endpoint - Implementat în FastAPI
✅ `PORT` dinamic - Configurat în app/core/config.py
✅ Căi relative - Actualizate
✅ Modele ML - În folder `models/` cu căi relative

---

## 🚀 Deploy complet în Railway

Proiectul folosește două servicii Railway din același repository:

| Serviciu | Dockerfile | Rol |
|-----------|------------|-----|
| `backend` | `Dockerfile` | API FastAPI: predict, train, anomaly, chat |
| `streamlit` | `Dockerfile.streamlit` | Interfața web pentru utilizator |

### 1. Publică repository-ul

```bash
git push origin main
```

### 2. Configurează serviciul backend

1. În proiectul Railway, creează sau păstrează serviciul GitHub existent.
2. La **Settings → Build**, setează Dockerfile Path la `Dockerfile`.
3. La **Variables**, setează `SUPABASE_URL` și `SUPABASE_SERVICE_ROLE_KEY`.
4. La **Networking**, generează un domeniu public.

### 3. Creează serviciul Streamlit

1. În același proiect Railway selectează **New → GitHub Repo** și alege același repository.
2. La **Settings → Build**, setează Dockerfile Path la `Dockerfile.streamlit`.
3. La **Variables**, setează:

```text
BACKEND_BASE_URL=https://ai-senzor-de-calitate-a-aerului-production.up.railway.app
SUPABASE_URL=<aceeași valoare ca backend-ul>
SUPABASE_SERVICE_ROLE_KEY=<aceeași valoare ca backend-ul>
```

4. La **Networking**, generează un domeniu public. Acesta este linkul pe care îl folosești pentru interfața Streamlit.

După fiecare push pe `main`, Railway va redeploya automat ambele servicii.

---

## Serviciu opțional: Ollama (chatbot cu LLM real în producție)

Local, chatbot-ul folosește Ollama de pe `127.0.0.1:11434`. Pe Railway acel Ollama local nu e
accesibil, de-aia `.env.railway` are `CHATBOT_USE_OLLAMA=false` și chatbot-ul cade pe
răspunsurile bazate pe reguli. Ca să ai același LLM și în producție:

1. În același proiect Railway, creează un al treilea serviciu **New → GitHub Repo** (același
   repo), cu Dockerfile Path setat la `Dockerfile.ollama`.
2. Nu genera domeniu public pentru acest serviciu — rămâne accesibil doar intern, prin
   rețeaua privată Railway (`<nume-serviciu>.railway.internal`).
3. Pe planul **Hobby** (8GB RAM/serviciu), modelul implicit din `Dockerfile.ollama`,
   `qwen2.5:7b-instruct` (același ca local), încape ca memorie. Inferența e însă pe CPU
   (fără GPU pe Hobby), deci răspunsurile vor fi mai lente decât local, iar rularea
   continuă a modelului consumă din creditul lunar de $5 al planului. Dacă e prea lent
   sau depășești creditul, schimbă `OLLAMA_MODEL` (din acel serviciu) cu un model mai mic,
   ex: `qwen2.5:3b-instruct`.
4. Adaugă un **Volume** montat pe `/root/.ollama` pentru serviciul Ollama, ca modelul
   descărcat să nu se piardă la fiecare redeploy (altfel se re-descarcă de fiecare dată,
   ceea ce încetinește pornirea).
5. Pe serviciul `backend`, la **Variables**, setează:

```text
CHATBOT_USE_OLLAMA=true
OLLAMA_BASE_URL=http://<nume-serviciu-ollama>.railway.internal:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

Folosește exact numele serviciului Ollama din Railway în loc de `<nume-serviciu-ollama>`.

---

## Deploy API separat

### Opțiunea A: Deploy via GitHub (Recomandată)

#### 1. Pregărează Repository-ul
```bash
# Asigură-te că ai comitat toate schimbările
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

#### 2. Conectează Railway cu GitHub
1. Mergi la [railway.app](https://railway.app)
2. Login cu GitHub account
3. Click "New Project" → "Deploy from GitHub repo"
4. Selectează repository-ul tău
5. Railway va detecta automat `Dockerfile`

#### 3. Configurează Environment Variables
În panelul Railway, adaugă sub "Variables":

**REQUIRED:**
```
SUPABASE_URL = your-supabase-url
SUPABASE_SERVICE_ROLE_KEY = your-service-role-key
```

**OPTIONAL (dacă vrei Ollama chatbot):**
```
CHATBOT_USE_OLLAMA = false
OLLAMA_BASE_URL = http://your-ollama-instance:11434
OLLAMA_MODEL = qwen2.5:7b-instruct
```

#### 4. Deploy
- Railway va detecta Dockerfile și va build + deploy automat
- Urmărește logs în Railway dashboard

---

### Opțiunea B: Deploy Local (Testing)

#### 1. Build Docker Image
```bash
docker build -t air-quality-api:latest .
```

#### 2. Test Local
```bash
docker run -p 8000:8000 \
  -e SUPABASE_URL="your-url" \
  -e SUPABASE_SERVICE_ROLE_KEY="your-key" \
  -e PORT=8000 \
  air-quality-api:latest
```

#### 3. Verific Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","service":"air-quality-ai-api"}
```

---

## 🔍 Verific Post-Deploy

### 1. Health Endpoint
```bash
curl https://your-railway-url.railway.app/health
```

### 2. Root Endpoint
```bash
curl https://your-railway-url.railway.app/
```

### 3. Swagger UI
Accesează: `https://your-railway-url.railway.app/docs`

### 4. Predicție Test
```bash
curl -X POST "https://your-railway-url.railway.app/predict" \
  -H "Content-Type: application/json"
```

---

## 📁 Structură Finală Proiect

```
.
├── app/
│   ├── __init__.py
│   ├── api/
│   │   └── main.py              ← Entry point API (FastAPI)
│   ├── core/
│   │   ├── config.py            ← Configurație (PORT dinamic)
│   │   └── database.py          ← Conexiune Supabase
│   ├── models/
│   │   ├── model_manager.py
│   │   ├── train_model.py
│   │   └── xgboost_model.py
│   └── services/
│       ├── anomaly_detector.py
│       ├── chatbot.py
│       └── predictor.py
│
├── models/                       ← Modele ML (încărcare relativă)
│   ├── air_quality_rf.pkl
│   ├── air_quality_svm.pkl
│   ├── air_quality_if.pkl
│   ├── xgboost.pkl
│   └── air_quality_model.pkl
│
├── .env.railway                 ← Template variabile Railway
├── .dockerignore               ← Exclude din Docker
├── Dockerfile                  ← Build container
├── railway.toml                ← Config Railway
├── railway.json                ← Config alternativă
├── requirements.txt            ← Dependințe (actualizat)
└── README.md
```

---

## ⚙️ Variabile de Mediu Railway

**REQUIRED:**
| Variabilă | Descriere | Exemplu |
|-----------|-----------|---------|
| `SUPABASE_URL` | URL Supabase | `https://xxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Cheie serviciu Supabase | `sb_secret_...` |

**OPTIONAL:**
| Variabilă | Descriere | Default |
|-----------|-----------|---------|
| `CHATBOT_USE_OLLAMA` | Activează Ollama | `false` |
| `OLLAMA_BASE_URL` | URL Ollama | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Model Ollama | `qwen2.5:7b-instruct` |
| `CHATBOT_ENABLE_WEB_SEARCH` | Web search | `true` |
| `PORT` | Port (auto-set) | `8000` |
| `API_HOST` | Host binding | `0.0.0.0` |

---

## 🔧 Troubleshooting

### Port 8000 Already in Use (Local)
```bash
# Schimbă PORT
docker run -p 9000:8000 -e PORT=8000 air-quality-api:latest
```

### Health Check Fails
- Verifică Supabase credentials
- Verific logs în Railway: `railway logs`
- Asigură-te că PORT env variable e setat

### Models Not Found
- Verific că models/ folder este în .dockerignore (NU E!)
- Verific path-uri relative în config.py

---

## 📊 Monitoring Railway

1. **Logs**: Railway Dashboard → Logs tab
2. **Metrics**: CPU, RAM, Network usage
3. **Deployments**: Istoric tuturor build-urilor

---

## 🎯 Checklist Final

- [x] requirements.txt fără Streamlit
- [x] Dockerfile cu multi-stage build
- [x] railway.toml și railway.json
- [x] .env.railway cu variabile necesare
- [x] config.py cu PORT dinamic
- [x] /health endpoint activ
- [x] Modele în folder relativă
- [x] Fără căi absolute Windows
- [x] .dockerignore completu
- [x] App testabilă în container

**Status: ✅ Ready for Railway Deployment**
