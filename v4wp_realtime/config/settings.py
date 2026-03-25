"""환경변수 기반 설정 로딩 (.env 파일 자동 로드)"""
import os
import json
from pathlib import Path

# .env 파일 자동 로드
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 미설치 시 os.environ만 사용


# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REALTIME_ROOT = Path(__file__).resolve().parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / 'data'
DB_PATH = DATA_DIR / 'v4wp.db'
SIGNALS_JSON = DATA_DIR / 'signals_history.json'
CACHE_DIR = PROJECT_ROOT / 'cache'

# Watchlist
WATCHLIST_PATH = REALTIME_ROOT / 'config' / 'watchlist.json'


def load_watchlist():
    with open(WATCHLIST_PATH, 'r') as f:
        return json.load(f)


# 섹터 → ETF 매핑 (유사 시그널 검색에서 시장/섹터 컨텍스트용)
SECTOR_ETF_MAP = {
    "Tech": "XLK",
    "Growth": "IWO",
    "Fintech": "XLF",
    "Quantum": "XLK",
    "Space": "ITA",
    "Benchmark": None,
    "Index": None,
}

# API Keys (환경변수에서 로딩)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '') or os.environ.get('ANTHROPIC_API_KEY', '')

# Telegram Mini App 설정
# BOT_USERNAME: BotFather에서 설정한 봇 유저네임 (@ 제외)
# 예: TELEGRAM_BOT_USERNAME=v4wp_bot
TELEGRAM_BOT_USERNAME = os.environ.get('TELEGRAM_BOT_USERNAME', '')
# Mini App 단축이름 (BotFather /newapp에서 설정)
TELEGRAM_MINIAPP_SHORT = os.environ.get('TELEGRAM_MINIAPP_SHORT', 'Dashboard')
# Mini App 직접 URL (GitHub Pages)
TELEGRAM_WEBAPP_URL = os.environ.get('TELEGRAM_WEBAPP_URL', '')
