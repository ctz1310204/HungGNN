#!/bin/bash
# Script để tạo repo MỚI và push code với citation đầy đủ

echo "=== TẠO REPO MỚI VỚI CITATION ==="
echo ""
echo "Nhập GitHub username của bạn:"
read USERNAME

echo ""
echo "Nhập tên repo mới (ví dụ: GNN_LSAP_modified):"
read REPO_NAME

echo ""
echo "✓ Bạn sẽ tạo: https://github.com/$USERNAME/$REPO_NAME"
echo ""

# Thêm citation vào README nếu chưa có
if ! grep -q "Original Repository" README.md 2>/dev/null; then
    echo ""
    echo "📝 Thêm citation vào README..."
    cat CITATION_TEMPLATE.md README.md > README_new.md
    mv README_new.md README.md
    git add README.md
    git commit -m "Add citation to original work"
    echo "✅ Đã thêm citation"
fi

echo ""
echo "🌐 BÂY GIỜ:"
echo "1. Vào https://github.com/new"
echo "2. Tạo repo: $REPO_NAME"
echo "3. Chọn PUBLIC (để dùng trên Colab)"
echo "4. KHÔNG tick 'Initialize with README'"
echo ""
echo "Đã tạo xong repo trên GitHub chưa? (y/n)"
read READY

if [ "$READY" != "y" ]; then
    echo "❌ Hủy. Chạy lại script khi đã tạo repo trên GitHub."
    exit 1
fi

# Setup remote và push
echo ""
echo "🚀 Đang setup remote và push..."

# Đổi origin từ aircarlo
git remote set-url origin https://github.com/$USERNAME/$REPO_NAME.git

# Push
git push -u origin main

echo ""
echo "✅ HOÀN TẤT!"
echo ""
echo "📍 Repo của bạn: https://github.com/$USERNAME/$REPO_NAME"
echo "📍 Clone trên Colab: !git clone https://github.com/$USERNAME/$REPO_NAME.git"
echo ""
echo "⚠️  QUAN TRỌNG: Kiểm tra README có citation chưa!"
