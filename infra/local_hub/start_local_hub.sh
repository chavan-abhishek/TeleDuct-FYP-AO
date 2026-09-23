#!/usr/bin/env bash
set -e

echo "=================================================="
echo "🚀 Starting TeleDuct Local Ingestion Hub..."
echo "=================================================="
cd "$(dirname "$0")"
docker compose up -d

echo "⏳ Waiting for PostgreSQL health..."
until docker exec teleduct-postgres pg_isready -U teleduct_admin -d teleduct_telemetry > /dev/null 2>&1; do
    sleep 1
done

echo "✅ Local Ingestion Hub is LIVE!"
echo "   - PostgreSQL: localhost:5432 (DB: teleduct_telemetry, User: teleduct_admin)"
echo "   - OTel HTTP:  http://localhost:4318"
echo "   - Prometheus: http://localhost:9090"
echo "   - Grafana:    http://localhost:3000 (admin / admin)"
echo ""
echo "👉 Start your ngrok tunnels in separate terminal windows:"
echo "   Terminal 1: ngrok http 4318"
echo "   Terminal 2: ngrok tcp 5432"