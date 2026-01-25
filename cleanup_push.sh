#!/bin/bash
cd /Users/vubeo/Documents/GIT_REPO/GNN_LSAP

# Abort any pending merge
git merge --abort 2>/dev/null || true

# Force push to remove docs and experiments from GitHub
git push origin main --force

echo ""
echo "✅ HOÀN TẤT!"
echo "📍 Kiểm tra: https://github.com/ctz1310204/HungGNN"
echo ""
echo "Folders đã xóa khỏi GitHub:"
echo "  - docs/"
echo "  - experiments/"
echo ""
echo "Folders vẫn còn ở local:"
ls -ld docs experiments 2>/dev/null || echo "  (đã bị xóa local)"
