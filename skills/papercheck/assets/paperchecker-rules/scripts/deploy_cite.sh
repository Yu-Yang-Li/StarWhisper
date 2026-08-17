#!/bin/bash
# 部署 TaShan-PaperChecker 到 cite.tashan.chat
# 架构：aup-test-01 (port 3950) ← autossh tunnel → ECS (port 13950) ← nginx ← 用户浏览器
#
# 用法：
#   bash scripts/deploy_cite.sh          # 完整部署（打包 → 上传 → 启动）
#   bash scripts/deploy_cite.sh backend  # 仅重启后端
#   bash scripts/deploy_cite.sh tunnel   # 仅重建隧道
#   bash scripts/deploy_cite.sh nginx    # 仅重载 nginx

set -e

AUP_HOST="aup-server"          # ~/.ssh/config 中的别名
ECS_HOST="tashan-ecs"          # ~/.ssh/config 中的别名
AUP_PORT=3950
ECS_PORT=13950
REMOTE_DIR="/home/aup/tashan-paperchecker"
TUNNEL_KEY="$HOME/.ssh/tashan_tunnel"
DOMAIN="cite.tashan.chat"
LOG="/tmp/tashan-paperchecker.log"

STEP="${1:-all}"

# ── 1. 打包 ──────────────────────────────────────────────────────────────
pack() {
  echo "==> 打包..."
  tar czf /tmp/tashan-paperchecker.tar.gz \
    --exclude="**/__pycache__" \
    --exclude="**/.DS_Store" \
    --exclude="**/.git" \
    --exclude="**/node_modules" \
    --exclude="temp_uploads/*" \
    --exclude="data/*" \
    --exclude="frontend/node_modules" \
    --exclude="frontend/dist" \
    -C "$(dirname "$0")/.." \
    app core contracts services config auth api utils front/tashan-ui front/web \
    run_server.py requirements.txt
  echo "    -> /tmp/tashan-paperchecker.tar.gz"
}

# ── 2. 上传 & 解压 ───────────────────────────────────────────────────────
upload() {
  echo "==> 上传到 aup..."
  ssh "$AUP_HOST" "mkdir -p $REMOTE_DIR && rm -rf $REMOTE_DIR/*"
  cat /tmp/tashan-paperchecker.tar.gz | ssh "$AUP_HOST" "cd $REMOTE_DIR && tar xzf -"
  echo "    -> $REMOTE_DIR"
}

# ── 3. 安装依赖 ──────────────────────────────────────────────────────────
install_deps() {
  echo "==> 安装 Python 依赖..."
  ssh "$AUP_HOST" "pip3 install --break-system-packages \
    fastapi uvicorn python-multipart pymupdf python-docx"
}

# ── 4. 启动后端 ──────────────────────────────────────────────────────────
start_backend() {
  echo "==> 启动后端 (port $AUP_PORT)..."
  # 写脚本文件到 aup，避免 ssh+& 的 hang 问题
  ssh "$AUP_HOST" "cat > /tmp/start_tashan_paperchecker.sh << 'EOF'
#!/bin/bash
cd $REMOTE_DIR
pkill -f 'tashan-paperchecker.*run_server' 2>/dev/null
pkill -f \"python3.*run_server\" 2>/dev/null
sleep 1
mkdir -p data temp_uploads
export SERVER_HOST=0.0.0.0
export SERVER_PORT=$AUP_PORT
export SERVER_RELOAD=false
setsid nohup python3 run_server.py > $LOG 2>&1 < /dev/null &
echo \"pid=\$!\"
sleep 4
if curl -sS --max-time 5 http://127.0.0.1:$AUP_PORT/api/health > /dev/null; then
  echo 'backend healthy'
  curl -sS http://127.0.0.1:$AUP_PORT/api/health
else
  echo 'backend NOT healthy'
  tail -30 $LOG
fi
EOF
chmod +x /tmp/start_tashan_paperchecker.sh"
  ssh "$AUP_HOST" "bash /tmp/start_tashan_paperchecker.sh"
}

# ── 5. autossh 隧道 ──────────────────────────────────────────────────────
start_tunnel() {
  echo "==> 建立 autossh 隧道 (ECS:$ECS_PORT → aup:$AUP_PORT)..."
  ssh "$AUP_HOST" "cat > /tmp/start_tashan_paperchecker_tunnel.sh << 'EOF'
#!/bin/bash
pkill -f \"autossh.*$ECS_PORT\" 2>/dev/null
sleep 1
setsid nohup autossh -M 0 -N \\
  -o 'ServerAliveInterval 30' -o 'ServerAliveCountMax 3' -o 'ExitOnForwardFailure yes' \\
  -i $TUNNEL_KEY \\
  -R ${ECS_PORT}:127.0.0.1:${AUP_PORT} \\
  root@101.200.234.115 > /tmp/tashan-paperchecker-tunnel.log 2>&1 < /dev/null &
echo \"autossh pid=\$!\"
sleep 4
if pgrep -f \"autossh.*$ECS_PORT\" > /dev/null; then
  echo 'tunnel running'
else
  echo 'tunnel FAILED'
  cat /tmp/tashan-paperchecker-tunnel.log
fi
EOF
chmod +x /tmp/start_tashan_paperchecker_tunnel.sh"
  ssh "$AUP_HOST" "bash /tmp/start_tashan_paperchecker_tunnel.sh"
}

# ── 6. nginx 配置 ────────────────────────────────────────────────────────
setup_nginx() {
  echo "==> 配置 nginx ($DOMAIN)..."
  ssh "$ECS_HOST" "cat > /etc/nginx/sites-enabled/$DOMAIN << 'NGINX_EOF'
server {
    listen 443 ssl;
    server_name $DOMAIN;
    ssl_certificate     /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    client_max_body_size 60M;

    location /api/ {
        proxy_pass http://127.0.0.1:$ECS_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
    }

    location / {
        proxy_pass http://127.0.0.1:$ECS_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
    }
}
server {
    listen 80;
    server_name $DOMAIN;
    return 301 https://\$host\$request_uri;
}
NGINX_EOF
nginx -t && systemctl reload nginx"
  echo "==> 验证..."
  ssh "$ECS_HOST" "curl -sS --max-time 5 -o /dev/null -w 'ECS→aup health HTTP %{http_code}\n' http://127.0.0.1:$ECS_PORT/api/health || true"
}

# ── 7. 签证书（首次部署） ─────────────────────────────────────────────────
issue_cert() {
  echo "==> 签自签证书（仅首次）..."
  ssh "$ECS_HOST" "
    mkdir -p /etc/letsencrypt/live/$DOMAIN
    if [ ! -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem ]; then
      openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/letsencrypt/live/$DOMAIN/privkey.pem \
        -out /etc/letsencrypt/live/$DOMAIN/fullchain.pem \
        -subj '/CN=$DOMAIN'
      echo '证书已生成'
    else
      echo '证书已存在，跳过'
    fi"
}

# ── 执行 ─────────────────────────────────────────────────────────────────
case "$STEP" in
  backend)
    start_backend
    ;;
  tunnel)
    start_tunnel
    ;;
  nginx)
    setup_nginx
    ;;
  all)
    pack
    upload
    install_deps
    start_backend
    start_tunnel
    issue_cert
    setup_nginx
    echo ""
    echo "==> 部署完成"
    echo "    https://$DOMAIN/ui/"
    ;;
  *)
    echo "用法: $0 [all|backend|tunnel|nginx]"
    exit 1
    ;;
esac
