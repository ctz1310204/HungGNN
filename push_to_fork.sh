#!/bin/bash
echo "Nhập GitHub username của bạn:"
read USERNAME

echo ""
echo "✓ Sẽ đổi remote đến: https://github.com/$USERNAME/GNN_LSAP"
echo ""

# Đổi remote từ aircarlo sang username của bạn
git remote set-url origin https://github.com/$USERNAME/GNN_LSAP.git

# Thêm upstream
git remote add upstream https://github.com/aircarlo/GNN_LSAP.git

echo "✓ Đã đổi remote!"
echo ""
echo "Remote hiện tại:"
git remote -v

echo ""
echo "Bạn có muốn push ngay không? (y/n)"
read CONFIRM

if [ "$CONFIRM" = "y" ]; then
    echo "Pushing to your fork..."
    git push -u origin main
    echo ""
    echo "✅ HOÀN TẤT! Kiểm tra tại: https://github.com/$USERNAME/GNN_LSAP"
else
    echo "Chưa push. Chạy lệnh này khi muốn push:"
    echo "  git push -u origin main"
fi
