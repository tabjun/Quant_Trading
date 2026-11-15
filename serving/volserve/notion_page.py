"""공개 집계를 노션 페이지 본문으로 쓴다(노션에서 "웹에 게시"하면 notion.site로 공개된다).

    blocks = build_blocks(public_summary(run))
    state = publish(blocks, token, page_id, old_block_ids=state.get("block_ids", []))

토큰은 노션 내부 연결(Developer connections)의 API 토큰이고, 게시할 페이지의 콘텐츠 사용 권한에 그 연결을 추가해 둔다.
새 블록을 먼저 붙이고 지난 블록을 지워서 갱신 중에 빈 페이지가 보이지 않게 한다.
"""
from __future__ import annotations

import time

from .agent import check_public, horizon_word

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
SCOPES = ("Q1", "Q2", "Q3", "Q4", "Q5")


def _rt(text: str) -> list[dict]:
    return [{"type": "text", "text": {"content": str(text)[:2000]}}]


def paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}


def heading(text: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rt(text)}}


def table(header: list[str], rows: list[list]) -> dict:
    def row(cells):
        return {"object": "block", "type": "table_row", "table_row": {"cells": [_rt(c) for c in cells]}}
    return {"object": "block", "type": "table", "table": {"table_width": len(header), "has_column_header": True,
                                                          "has_row_header": True, "children": [row(header)] + [row(r) for r in rows]}}


def build_blocks(summary: dict) -> list[dict]:
    rt = summary["route_table"]
    blocks = [
        paragraph(f"기준 시각 {summary['as_of']} · {summary['tickers']}종목 × {len(summary['horizons_min'])}구간. 화면에는 집계만 싣습니다."),
        heading("에이전트 단계(최근 실행 시간, 초)"),
        table(["단계", "초"], [[k, f"{v:.2f}"] for k, v in summary["stage_seconds"].items()]),
        heading("라우팅 표: 구간 × 국면별 1위 모델"),
        table(["구간", *SCOPES], [[horizon_word(int(h))] + [f"{row[q]['model']} (구분 안 됨 {row[q]['ties'] if row[q]['ties'] is not None else '미상'})"
                                                         for q in SCOPES] for h, row in rt.items()]),
        heading("최근 실행: 구간별 국면 분포(종목 수)"),
        table(["구간", *SCOPES], [[horizon_word(int(h))] + [c[q] for q in SCOPES] for h, c in summary["regime_counts"].items()]),
        heading("최근 실행: 쓴 모델(칸 수)"),
        table(["구간", "모델별 칸 수"], [[horizon_word(int(h)), ", ".join(f"{m} {n}" for m, n in ms.items())]
                                     for h, ms in summary["model_counts"].items()]),
    ]
    check_public(blocks)
    return blocks


def _call(session, method: str, path: str, body: dict | None = None) -> dict:
    for attempt in range(4):
        r = session.request(method, f"{API}{path}", json=body, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        r.raise_for_status()
        time.sleep(0.35)  # 노션 API 평균 초당 3회 한도 안쪽
        return r.json()
    raise RuntimeError(f"노션 API {method} {path}: 재시도 뒤에도 실패")


def publish(blocks: list[dict], token: str, page_id: str, old_block_ids: list[str] = ()) -> dict:
    import requests

    with requests.Session() as s:
        s.headers.update({"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION})
        got = _call(s, "PATCH", f"/blocks/{page_id}/children", {"children": blocks})
        for bid in old_block_ids:
            _call(s, "DELETE", f"/blocks/{bid}")
    return {"page_id": page_id, "block_ids": [b["id"] for b in got["results"]]}
