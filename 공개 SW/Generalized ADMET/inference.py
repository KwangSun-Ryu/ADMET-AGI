#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 파일: OpenSearch 기반 단순 RAG + OpenAI Chat Completions(function calling)
- single turn(사용자 입력 1회)이지만, 모델은 도구를 여러 번 호출할 수 있도록 반복 추론 지원
- RAG 도구: rag_search (semantic-only, OpenSearch kNN 사용)
- rag_search 입력: query (string) 하나만
- rag_search 출력: OpenSearch 검색 응답 전체(JSON)에서 각 hit._source의 'embedding'만 제거
"""

import os
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import requests
import openai

try:
    from opensearchpy import OpenSearch
except ImportError as e:
    raise SystemExit("opensearch-py가 설치되어 있지 않습니다. `pip install opensearch-py` 후 실행하세요.") from e


# =========================
# 0) 환경 설정 / 기본값
# =========================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "none")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "text-embedding-3-large")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "none")

OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_INDEX = os.getenv("OS_INDEX", "rag-index")

# 인덱스 매핑에 맞게 필드명 맞춰주세요.
OS_EMBED_FIELD = os.getenv("OS_EMBED_FIELD", "embedding")  # 벡터 필드 (k-NN)

# k-NN 검색 상수
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# 도구 반복 호출 상한 (무한루프 방지)
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "6"))


# =========================
# 1) Tool Spec 빌더 (요구 스펙 유지)
# =========================

def _build_fn_spec_from_raw(raw_tool):
    """MCP raw_tool → OpenAI function spec (요청된 스펙 그대로)"""
    schema = getattr(raw_tool, "inputSchema", {}) or {}
    props = {}
    for prop_name, prop_schema in schema.get("properties", {}).items():
        desc = prop_schema.get("description", prop_schema.get("title", "")).strip()
        props[prop_name] = {
            "type": prop_schema.get("type", "string"),
            "description": desc
        }
    required = schema.get("required", [])
    return {
        "type": "function",
        "function": {
            "name": raw_tool.name,
            "description": (raw_tool.description or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        }
    }


# =========================
# 2) LLM 파이프라인 (단순화 버전)
#    - OpenSearch 클라이언트/임베딩 호출/툴 호출을 한 클래스에 통합
# =========================

class LLMPipeline:
    def __init__(self, model_url: str, model_name: str):
        # OpenAI
        openai.api_key = OPENAI_API_KEY
        openai.base_url = model_url
        self.model_name = model_name

        # OpenSearch (로그인/SSL 없이 고정)
        self.os_client = OpenSearch(
            hosts=[{"host": OS_HOST, "port": OS_PORT}],
            use_ssl=False,
            http_compress=True,
        )
        # 인덱스 확인
        try:
            if not self.os_client.indices.exists(index=OS_INDEX):
                raise RuntimeError(f"OpenSearch 인덱스가 존재하지 않습니다: {OS_INDEX}")
        except Exception as e:
            raise SystemExit(f"OpenSearch 연결/인덱스 확인 실패: {e}") from e

        # 간단 RawTool → tool spec
        raw_tool = SimpleNamespace(
            name="rag_search",
            description=(
                "간단한 RAG 검색 도구입니다. "
                "임베딩+k-NN(Vector)로 OpenSearch 인덱스를 조회하고, "
                "검색 응답에서 각 문서의 'embedding' 필드를 제거한 원본 JSON을 반환합니다."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "검색 질의(필수). 3?8 토큰의 핵심 키워드를 추천합니다. "
                            "예: \"2024 성과계획서\" \"프로그램목표 1-1\""
                        ),
                    }
                },
                "required": ["query"]
            }
        )
        self.tool_spec = _build_fn_spec_from_raw(raw_tool)
        self._tool_impl_map = {"rag_search": self._call_rag_search}

    # OpenAI Chat 호출
    def _chat_once(self, messages: list, tools: list):
        return openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            tools=tools,
            tool_choice="auto",
            stream=False
        )
    
    def _embed_query(self, query: str) -> List[float]:
        body = {"texts": [query], "batch_size": 64, "normalize": True}
        response = requests.post(url=EMBEDDING_URL, json=body)
        if response.status_code == 200:
            result = response.json()
            return result.get("embeddings", [[0.0] * 768])[0]
        return [0.0] * 768  # 안전 가드

    # rag_search 구현: query → embed → kNN → embedding 필드 제거 → JSON 문자열 반환
    def _call_rag_search(self, args: Dict[str, Any]) -> str:
        query = (args or {}).get("query")
        if not query or not str(query).strip():
            return json.dumps({"error": "query is required"}, ensure_ascii=False)

        # 1) 임베딩
        try:
            emb = self._embed_query(query)
        except Exception as e:
            return json.dumps({"error": f"embedding failed: {e}"}, ensure_ascii=False)

        # 2) OpenSearch k-NN (BM25 없음)
        body = {
            "size": DEFAULT_TOP_K,
            "query": {
                "knn": {
                    OS_EMBED_FIELD: {           # 예: "embedding"
                        "vector": emb,          # 리스트[float] 그대로
                        "k": DEFAULT_TOP_K
                    }
                }
            }
        }
        try:
            resp = self.os_client.search(index=OS_INDEX, body=body)
        except Exception as e:
            return json.dumps({"error": f"opensearch knn search failed: {e}"}, ensure_ascii=False)

        # 3) embedding 필드 제거
        print(f'opensearch_resp: {resp}')
        try:
            filtered = json.loads(json.dumps(resp, ensure_ascii=False))  # deep copy
            hits = filtered.get("hits", {}).get("hits", [])
            for h in hits:
                src = h.get("_source", {})
                if isinstance(src, dict) and OS_EMBED_FIELD in src:
                    src.pop(OS_EMBED_FIELD, None)
        except Exception:
            filtered = {"error": "unexpected opensearch response structure"}

        return json.dumps(filtered, ensure_ascii=False)

    @staticmethod
    def _as_dict_message(msg_obj) -> Dict[str, Any]:
        if isinstance(msg_obj, dict):
            return msg_obj
        d = {
            "role": getattr(msg_obj, "role", "assistant"),
            "content": getattr(msg_obj, "content", None),
        }
        if getattr(msg_obj, "tool_calls", None):
            d["tool_calls"] = []
            for tc in msg_obj.tool_calls:
                d["tool_calls"].append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        return d

    # === public: single-turn 처리 메서드 (요청에 따라 이름 변경) ===
    def chat_process(self, user_input: str):
        system_prompt = (
            "당신은 필요한 경우 rag_search 도구를 여러 번 호출해 적절한 컨텍스트를 수집한 뒤 "
            "도구 응답(JSON) 내 정보를 근거로 한국어로 정확하고 간결하게 답변하세요. "
            "rag_search는 'query' 하나만 입력받고, OpenSearch 응답에서 'embedding' 필드를 제거한 원본 JSON을 반환합니다. "
            "최종적으로 사용자가 이해하기 쉬운 답변을 제시하세요."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        tools = [self.tool_spec]

        tool_calls_used = 0
        last_assistant_content: Optional[str] = None

        # 반복 루프: 모델이 도구를 0~N번 호출 가능
        while tool_calls_used <= MAX_TOOL_CALLS:
            resp = self._chat_once(messages=messages, tools=tools)
            msg = resp.choices[0].message
            print("\nAssistant:")
            print(msg.content or "")
            # 대화 로그에 assistant 메시지(도구 호출 의도 포함)를 추가
            messages.append(self._as_dict_message(msg))
            # 도구 호출 유무 확인
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                # 도구 호출이 없다면 이것이 최종 답변
                last_assistant_content = msg.get("content") if isinstance(msg, dict) else msg.content
                print("\nAssistant:")
                print(last_assistant_content or "")
                break
                

            # 도구 호출 수행
            for tc in tool_calls:
                if tc.type != "function":
                    continue
                fn_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                print(f'tool_call args :{args}')

                impl = self._tool_impl_map.get(fn_name)
                if impl is None:
                    tool_result = json.dumps({"error": f"no implementation for tool '{fn_name}'"}, ensure_ascii=False)
                else:
                    tool_result = impl(args)

                print(f'tool_result :{tool_result}')

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_result
                })

                tool_calls_used += 1
                if tool_calls_used >= MAX_TOOL_CALLS:
                    break

            if tool_calls_used >= MAX_TOOL_CALLS:
                # 상한 도달 시 마지막으로 'tool_choice=none'으로 강제 종료 답변 유도
                final = openai.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.95,
                    tools=tools,
                    tool_choice="none",
                    stream=False
                )
                last_assistant_content = final.choices[0].message.get("content")
                break

            print("\nAssistant:")
            print(last_assistant_content or "")
        



# =========================
# 3) main
# =========================

def main():
    # OpenAI 키 설정 알림
    openai.api_key = "none"

    # 파이프라인 준비
    pipe = LLMPipeline(model_url=OPENAI_BASE_URL, model_name=OPENAI_MODEL)

    # 사용자 입력 (single turn)
    user_input = input("User: ").strip()
    if not user_input:
        print("입력이 비어 있습니다. 프로그램을 종료합니다.")
        return

    # 실행 (chat_process)
    pipe.chat_process(user_input)


if __name__ == "__main__":
    main()
