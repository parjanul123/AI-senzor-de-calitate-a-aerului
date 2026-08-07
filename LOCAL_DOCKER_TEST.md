# Local Testing în Docker înainte de Railway Deploy

## 🐳 Build Local Docker Image

```bash
# Merge în folder proiect
cd "d:\AI senzor de temperatura"

# Build image (aceasta va lua câteva minute)
docker build -t air-quality-api:latest .

# Check dacă build-ul a reușit
docker images | grep air-quality-api
```

---

## 🧪 Test Local cu Variabile de Mediu

### Opțiunea 1: Linie de comandă (Simple)

```bash
docker run -d \
  --name air-quality-api-test \
  -p 8000:8000 \
  -e SUPABASE_URL="https://your-project.supabase.co" \
  -e SUPABASE_SERVICE_ROLE_KEY="your-key-here" \
  -e PORT=8000 \
  air-quality-api:latest
```

### Opțiunea 2: Din .env File (Recomandată)

1. **Creează `.env.local` cu valorile tale:**
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key-here
CHATBOT_USE_OLLAMA=false
PORT=8000
```

2. **Run container cu .env file:**
```bash
docker run -d \
  --name air-quality-api-test \
  -p 8000:8000 \
  --env-file .env.local \
  air-quality-api:latest
```

---

## ✅ Verificări Post-Start

### 1. Container Status
```bash
# Verific dacă containerul ruleaza
docker ps | grep air-quality-api-test

# Verific logs
docker logs air-quality-api-test

# Verific full logs (mai detaliat)
docker logs -f air-quality-api-test  # Press Ctrl+C to stop
```

### 2. Health Check Endpoint
```bash
# Cel mai simplu test
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","service":"air-quality-ai-api"}
```

### 3. Root Endpoint
```bash
curl http://localhost:8000/
```

### 4. Swagger UI
Deschide în browser: `http://localhost:8000/docs`

### 5. Test Predicție
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n"
```

---

## 🔧 Debugging în Container

### View Logs Real-time
```bash
docker logs -f air-quality-api-test
```

### Execute Commands în Container
```bash
# Connect la shell
docker exec -it air-quality-api-test /bin/bash

# Check Python version
docker exec air-quality-api-test python --version

# Check installed packages
docker exec air-quality-api-test pip list
```

### Check Port Bindings
```bash
# Verific ce porturi sunt ascultate în container
docker exec air-quality-api-test netstat -tuln
```

---

## 🧹 Cleanup

### Stop Container
```bash
docker stop air-quality-api-test
```

### Remove Container
```bash
docker rm air-quality-api-test
```

### Remove Image
```bash
docker rmi air-quality-api:latest
```

### Cleanup complet (ALL containers)
```bash
# Stop all
docker stop $(docker ps -q)

# Remove all stopped
docker container prune
```

---

## 🚨 Common Issues & Solutions

### Error: "bind: address already in use"
**Problemă:** Port 8000 e deja folosit
**Soluție:**
```bash
# Use different port
docker run -p 9000:8000 air-quality-api:latest

# Or kill procesul care îl folosește
lsof -i :8000
kill -9 <PID>
```

### Error: "Failed to connect to Supabase"
**Problemă:** Credențiale Supabase incorecte
**Soluție:**
```bash
# Check environment variables în container
docker exec air-quality-api-test env | grep SUPABASE

# Verific .env.local
cat .env.local
```

### Error: "Port variable not recognized"
**Problemă:** PORT env var nu e setat
**Soluție:** Asigură-te că ai `-e PORT=8000` în docker run

### Logs show "ModuleNotFoundError"
**Problemă:** Dependență lipsă
**Soluție:**
```bash
# Rebuild image (poate cache o versiune veche)
docker build --no-cache -t air-quality-api:latest .
```

---

## 📊 Performance Test

### Load Test (dacă vrei să simulezi trafic)

```bash
# Install Apache Bench (dacă nu e instalat)
# Windows: choco install apache-bench
# Linux: sudo apt-get install apache2-utils

# 100 requests, 10 concurrent
ab -n 100 -c 10 http://localhost:8000/health

# Rezultat exemplu:
# Requests per second: 250.42 [#/sec]
# Time per request: 3.993 [ms]
```

---

## ✅ Validation Checklist

- [ ] Docker build reușit (`Successfully tagged...`)
- [ ] Container pornit fără erori
- [ ] `/health` endpoint răspunde cu {"status":"ok"}
- [ ] `/docs` accessible în browser  
- [ ] Supabase connection initialized (check logs)
- [ ] Modele ML loaded (check logs)
- [ ] Fără liniile de erori în logs
- [ ] Port 8000 e accessible

---

## 🎯 După Validare Locală

Dacă toate verificările trec ✅:

1. **Push la GitHub**
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **Deploy pe Railway**
   - Mergi la railway.app
   - Conectează GitHub repository
   - Railway va build automat din Dockerfile
   - Adaugă Supabase credentials în Railway dashboard
   - Deploy e live! 🚀

---

**Status: Ready for Railway if local tests pass ✅**
