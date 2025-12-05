import sys
import os
import importlib.util

def check_package(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"❌ {name} NOT installed")
        return False
    else:
        print(f"✅ {name} installed")
        return True

print("Verifying News Pipeline Dependencies...")
packages = ["fastapi", "uvicorn", "transformers", "torch", "bs4", "dotenv"]
all_good = True
for p in packages:
    if not check_package(p):
        all_good = False

if all_good:
    print("\n🎉 All News Pipeline dependencies are ready!")
    print("You can now run: run_news_system.bat")
else:
    print("\n⚠️ Some packages are missing. Run: pip install -r requirements_news.txt")
