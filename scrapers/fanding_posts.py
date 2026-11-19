"""팬딩 크리에이터의 공개 포스트 목록을 JSONL로 저장한다.

로그인 없이 누구나 보는 목록 정보(제목, 작성 시각, 조회/좋아요/댓글 수, 시리즈,
공개 범위)만 받는다. 본문과 댓글은 받지 않는다.

사용법:
    python scrapers/fanding_posts.py T3chfeed
    python scrapers/fanding_posts.py T3chfeed --out scraped/fanding --delay 1.5
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://fanding.kr/rest/post_list"
KST = timezone(timedelta(hours=9))
PAGE_SIZE = 24


def make_opener():
    ctx = ssl.create_default_context()
    ca = os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        ctx.load_verify_locations(ca)
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def fetch_page(opener, creator, last_post_no=None):
    params = {
        "sMemberUrl": creator,
        "iLimit": PAGE_SIZE,
        "sSortOrder": "recent",
        "sVisibleOnlyOption": "F",
    }
    if last_post_no:
        params["iLastPostNo"] = last_post_no
    req = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "Mozilla/5.0 (personal-research)", "Accept": "application/json"},
    )
    for attempt in range(4):
        try:
            with opener.open(req, timeout=30) as r:
                body = json.loads(r.read().decode("utf-8"))
            if not body.get("bIsResult"):
                raise RuntimeError(f"API error: {body.get('sCode')} {body.get('sMessage')}")
            return body["aData"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def to_record(p, creator, fetched_at):
    created_kst = datetime.strptime(p["sInsDatetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    auth = p.get("aAuthInfo") or {}
    return {
        "source": "fanding",
        "creator": creator,
        "post_no": p["iPostNo"],
        "url": f"https://fanding.kr/@{creator}/post/{p['iPostNo']}/",
        "title": p["sTitle"],
        "post_type": p.get("sType"),
        "content_type": p.get("sContentType"),
        "duration_sec": p.get("iDuration"),
        "series_no": p.get("iCollectionNo"),
        "series_title": p.get("sCollectionTitle"),
        "published_kst": created_kst.isoformat(),
        "published_utc": created_kst.astimezone(timezone.utc).isoformat(),
        "views": p.get("iViewCount"),
        "likes": p.get("iLikeCount"),
        "comments": p.get("iReplyCount"),
        "public_range": auth.get("sPublicRange"),
        "is_paid_content": p.get("sIsPaidContent") == "T",
        "thumbnail_url": p.get("sPostThumbnailUrl"),
        "fetched_at_utc": fetched_at,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("creator", help="팬딩 주소의 @ 뒤 이름 (예: T3chfeed)")
    ap.add_argument("--out", default="scraped/fanding")
    ap.add_argument("--delay", type=float, default=1.5, help="요청 사이 대기 초")
    ap.add_argument("--max-pages", type=int, default=0, help="0이면 끝까지")
    args = ap.parse_args()

    opener = make_opener()
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.creator}_posts_{stamp}.jsonl")

    seen, last_no, page, total = set(), None, 0, None
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            data = fetch_page(opener, args.creator, last_no)
            total = data.get("iTotalCount", total)
            posts = data.get("aPostList") or []
            new = 0
            for p in posts:
                if p["iPostNo"] in seen:
                    continue
                seen.add(p["iPostNo"])
                f.write(json.dumps(to_record(p, args.creator, fetched_at), ensure_ascii=False) + "\n")
                new += 1
            page += 1
            print(f"page {page}: +{new} (누적 {len(seen)}/{total})", file=sys.stderr)
            next_no = data.get("iLastPostNo")
            if not posts or new == 0 or not next_no or next_no == last_no:
                break
            if args.max_pages and page >= args.max_pages:
                break
            last_no = next_no
            time.sleep(args.delay)

    print(out_path)


if __name__ == "__main__":
    main()
