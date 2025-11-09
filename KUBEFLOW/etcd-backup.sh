#!/bin/bash
set -euo pipefail

HOSTNAME=$(hostname)
CA_CERT="/etc/ssl/etcd/ssl/ca.pem"
ETCD_CERT="/etc/ssl/etcd/ssl/node-${HOSTNAME}.pem"
ETCD_KEY="/etc/ssl/etcd/ssl/node-${HOSTNAME}-key.pem"
ENDPOINT="https://127.0.0.1:2379"
BACKUP_DIR="/backup/etcd"
RETAIN_DAYS=7

mkdir -p "${BACKUP_DIR}"

DATE=$(date +%Y%m%d-%H%M%S)
SNAPSHOT_FILE="${BACKUP_DIR}/etcd-snapshot-${DATE}.db"

echo "[$(date)] Starting etcd snapshot backup to ${SNAPSHOT_FILE}..."

ETCDCTL_API=3 etcdctl \
  --cacert="${CA_CERT}" \
  --cert="${ETCD_CERT}" \
  --key="${ETCD_KEY}" \
  --endpoints="${ENDPOINT}" \
  snapshot save "${SNAPSHOT_FILE}"

# 验证快照
ETCDCTL_API=3 etcdctl \
  --cacert="${CA_CERT}" \
  --cert="${ETCD_CERT}" \
  --key="${ETCD_KEY}" \
  --endpoints="${ENDPOINT}" \
  snapshot status "${SNAPSHOT_FILE}" >/dev/null

# 清理旧备份（保留7天）
find "${BACKUP_DIR}" -name "etcd-snapshot-*.db" -mtime +${RETAIN_DAYS} -delete

echo "[$(date)] Backup completed."
