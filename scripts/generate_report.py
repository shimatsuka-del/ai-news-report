#!/usr/bin/env python3
"""
AI News Report Generator
Google News RSS から記事を取得し、Gemini API で処理して HTML レポートを生成する。
"""

import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader

# ── 定数 ─────────────────────────────────────────────────────────────────────

JST = timezone(timedelta(hours=9))

KEYWORDS = [
    "ChatGPT 活用",
    "生成AI 業務効率化",
    "Claude AI",
    "Gemini AI",
    "中小企業 AI",
]

MAX_ARTICLES = 2
MAX_SEEN = 60
MODEL = "gemini-2.5-flash"

WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SEEN_FILE = ROOT_DIR / "seen_articles.json"
TEMPLATE_DIR = ROOT_DIR / "template"
DIST_DIR = ROOT_DIR / "dist"


# ── ユーティリティ ────────────────────────────────────────────────────────────

def format_date_ja(dt: datetime) -> str:
    wd = WEEKDAY_JA[dt.weekday()]
    return f"{dt.year}年{dt.month}月{dt.day}日（{wd}）"


def load_seen() -> list[str]:
    if SEEN_FILE.exists():
        with open(SEEN_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_seen(seen: list[str]) -> None:
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(seen[-MAX_SEEN:], f, ensure_ascii=False, indent=2)


def write_github_output(**kwargs) -> None:
    """GitHub Actions の GITHUB_OUTPUT ファイルに変数を書き込む。"""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def strip_html(text: str) -> str:
    """HTML タグを除去し、エンティティをデコードする。"""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


# ── RSS 取得 ──────────────────────────────────────────────────────────────────

def fetch_rss(keyword: str) -> list[dict]:
    """指定キーワードの Google News RSS を取得する。"""
    url = (
        f"https://news.google.com/rss/search"
        f"?q={quote(keyword)}&hl=ja&gl=JP&ceid=JP:ja"
    )
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        published_str = ""
        if getattr(entry, "published_parsed", None):
            dt = datetime(
                *entry.published_parsed[:6], tzinfo=timezone.utc
            ).astimezone(JST)
            published_str = f"{dt.year}年{dt.month}月{dt.day}日"

        articles.append({
            "title": strip_html(entry.get("title", "")),
            "url": entry.get("link", ""),
            "source": entry.get("source", {}).get("title", "不明"),
            "published": published_str,
            "description": strip_html(entry.get("summary", "")),
        })
    return articles


def fetch_all_articles() -> list[dict]:
    """全キーワードの RSS を取得し、URL 単位で重複除去して返す。"""
    all_articles: list[dict] = []
    seen_urls: set[str] = set()
    for keyword in KEYWORDS:
        try:
            arts = fetch_rss(keyword)
            for a in arts:
                if a["url"] and a["url"] not in seen_urls:
                    seen_urls.add(a["url"])
                    all_articles.append(a)
        except Exception as e:
            print(f"[warn] '{keyword}' の RSS 取得に失敗: {e}", file=sys.stderr)
    return all_articles


# ── Gemini API ────────────────────────────────────────────────────────────────

def call_gemini_json(
    client: genai.Client, prompt: str, max_retries: int = 3
) -> dict | None:
    """Gemini API を呼び出し、JSON をパースして返す。失敗時は None。"""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(
                f"[warn] Gemini 呼び出し失敗 (試行 {attempt + 1}/{max_retries}): {e}",
                file=sys.stderr,
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


def select_articles(client: genai.Client, articles: list[dict]) -> list[int]:
    """Gemini に記事リストを渡し、最大 MAX_ARTICLES 件のインデックスを返す。"""
    candidates = [
        {
            "index": i,
            "title": a["title"],
            "source": a["source"],
            "excerpt": a["description"][:200],
        }
        for i, a in enumerate(articles)
    ]
    prompt = f"""あなたはAIニュースのキュレーターです。
以下の記事リストから、「岐阜県の中小企業向けにAI活用研修を提供している講師」にとって\
最も価値ある記事を最大{MAX_ARTICLES}件選んでください。

選定基準：
- 中小企業でのAI活用に直接関係する事例・手法
- 業務効率化・コスト削減・生産性向上に関する実践的な内容
- AI研修で受講者が「すぐ使える」と感じられる内容
- 最新のAIツール・サービスの動向で研修に取り込める情報

記事リスト：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

以下の JSON 形式のみで回答してください（説明不要）：
{{"selected_indices": [0, 2]}}"""

    result = call_gemini_json(client, prompt)
    if result and "selected_indices" in result:
        indices = [
            i for i in result["selected_indices"] if 0 <= i < len(articles)
        ]
        return indices[:MAX_ARTICLES]
    # フォールバック: 先頭から MAX_ARTICLES 件
    return list(range(min(MAX_ARTICLES, len(articles))))


def process_article(client: genai.Client, article: dict) -> dict | None:
    """Gemini に記事を渡し、重要度・要約・活用ヒントを生成する。"""
    prompt = f"""以下のニュース記事を分析し、岐阜県の中小企業向けAI活用研修講師の視点から情報を生成してください。

タイトル: {article['title']}
ソース: {article['source']}
掲載日: {article['published']}
内容（抜粋）: {article['description'][:600]}

以下の JSON 形式のみで回答してください（説明不要）：
{{
  "importance": "high または mid のどちらか（high: 中小企業AI活用に直結する重要情報、mid: 参考になる情報）",
  "summary": "記事の要点を元記事未読でも理解できるよう説明する5〜8行相当のテキスト。句点（。）で区切り、自然な日本語で書くこと。",
  "hint": "この記事を踏まえて、岐阜県の中小企業向けAI活用研修にどう活かせるかの具体的な提案（5〜8行相当）。研修内容・受講者への伝え方・演習のアイデアなどを含めること。句点（。）で区切り、自然な日本語で書くこと。"
}}"""

    result = call_gemini_json(client, prompt)
    if not result or not all(k in result for k in ("importance", "summary", "hint")):
        return None

    # importance の正規化（Gemini が日本語で返す場合にも対応）
    importance_raw = str(result.get("importance", "mid")).lower()
    if importance_raw == "high" or "高" in importance_raw:
        importance = "high"
    else:
        importance = "mid"

    return {
        "importance": importance,
        "summary": result["summary"],
        "hint": result["hint"],
    }


# ── HTML 生成 ─────────────────────────────────────────────────────────────────

def render_html(articles_data: list[dict], report_date: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report-template.html")
    return template.render(report_date=report_date, articles=articles_data)


# ── メイン処理 ────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(JST)
    report_date = format_date_ja(now)
    surge_domain = os.environ.get(
        "SURGE_DOMAIN",
        f"ai-news-report-{now.strftime('%Y%m%d')}.surge.sh",
    )

    # ① 取得済み URL を読み込む
    seen = load_seen()
    seen_set = set(seen)

    # ② RSS 取得
    print("RSS フィードを取得中...")
    all_articles = fetch_all_articles()
    new_articles = [a for a in all_articles if a["url"] not in seen_set]
    print(f"合計 {len(all_articles)} 件取得、うち新規 {len(new_articles)} 件。")

    if not new_articles:
        print("新規記事なし。レポート生成をスキップします。")
        write_github_output(has_articles="false")
        sys.exit(0)

    # ③ Gemini クライアント初期化
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # ④ 記事を選定
    print("Gemini で記事を選定中...")
    selected_indices = select_articles(client, new_articles)
    selected = [new_articles[i] for i in selected_indices]
    print(f"{len(selected)} 件を選定しました。")

    # ⑤ 各記事を処理（要約・活用ヒント・重要度生成）
    processed: list[dict] = []
    for art in selected:
        print(f"処理中: {art['title'][:60]}...")
        result = process_article(client, art)
        if result:
            processed.append({
                "title": art["title"],
                "source": art["source"],
                "published": art["published"],
                "url": art["url"],
                "importance_label": "重要度：高" if result["importance"] == "high" else "重要度：中",
                "importance_class": "badge-high" if result["importance"] == "high" else "badge-mid",
                "summary": result["summary"],
                "hint": result["hint"],
            })
        else:
            print(f"[warn] 記事の処理に失敗しました: {art['title'][:40]}", file=sys.stderr)

    if not processed:
        print("すべての記事処理が失敗しました。レポート生成をスキップします。")
        write_github_output(has_articles="false")
        sys.exit(0)

    # ⑥ HTML 生成
    print("HTML を生成中...")
    html_content = render_html(processed, report_date)
    DIST_DIR.mkdir(exist_ok=True)
    output_path = DIST_DIR / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"HTML を書き出しました: {output_path}")

    # ⑦ 取得済み URL を更新
    new_urls = [a["url"] for a in selected]
    save_seen(seen + new_urls)
    print(f"seen_articles.json を更新しました（{len(seen) + len(new_urls)} 件）。")

    # ⑧ GitHub Actions 向け出力変数
    write_github_output(
        has_articles="true",
        surge_domain=surge_domain,
        report_date=report_date,
        article_count=str(len(processed)),
    )
    print(f"完了。デプロイ先: https://{surge_domain}")


if __name__ == "__main__":
    main()
