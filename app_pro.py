# Modified integrated Streamlit app with improved chat UX and auto-design behavior
# Filename: integrated_cake_studio_streamlit.py
# - Shows user's message immediately and displays an assistant "typing" placeholder
# - Uses GPT intent analysis BEFORE image generation so the generated design reflects updated design/lettering
# - If the user's message requests a design change + image generation, the new design is used

import os
import re
import json
import time
import base64
from io import BytesIO


import streamlit as st
from openai import BadRequestError, OpenAI, PermissionDeniedError

# =============================================================
#  CONFIG / DATA
# =============================================================
st.set_page_config(page_title="통합 커스텀 케이크 스튜디오", page_icon=":cake:", layout="wide")

API_KEY_FILE = "openai_key.txt"

MENU = {
    "sizes": {"1호": 25000, "2호": 36000, "3호": 47000, "하트": 42000},
    "fillings": {"생크림": 0, "초코": 3500, "레드벨벳": 6000, "티라미수": 5500},
    "base_custom": 20000,
    "extras": {
        "image": 10000, "color": 5000, "object": 2000, "long_lettering": 3000
    }
}

SCHEDULE = {
    "2025-12-24": ["10:00", "11:00", "14:00", "16:00"],
    "2025-12-25": [],
    "2025-12-26": ["11:00", "13:00", "15:00", "17:00", "19:00"]
}

# ✅ 현실적인, 심플한 디자인을 강하게 유도하는 시스템 프롬프트
DEFAULT_DESIGN_SYSTEM_PROMPT = (
    "너는 커스텀 케이크 디자이너야. 다음 규칙을 반드시 따르며 케이크를 디자인해줘:\n"
    "1) 결과물은 항상 \"현실적으로 제작 가능한 실제 1단(원단) 케이크\"여야 한다.\n"
    "2) 사용자의 설명은 반드시 케이크 디자인에 반영한다.\n"
    "3) 생성하는 이미지에는 케이크 이외의 부수적인 요소는 포함되면 안돼. \n"
    "4) 케이크는 과장되거나 비현실적인 형태(공중에 떠 있는 장식, 과도하게 큰 조형물, 지나치게 복잡한 구조)를 가지면 안 된다.\n"
    "5) 전체 분위기는 '심플하고 미니멀한 디자인'을 기본으로 하고, 색상은 최대 2~3가지 안에서 조합한다.\n"
    "6) 케이크 상단과 옆면 장식은 실제 동네 케이크 가게나 홈베이커가 구현할 수 있을 정도의 난이도로 제한한다.\n"
    "7) 출력은 한국어 bullet 형식 5줄 이내로 작성한다.\n"
    "8) 이미지/시안 생성 시 케이크는 반드시 단층(single-tier)으로 표현하고, 2단 이상은 절대 안 된다.\n"
    "9) 이미지/시안 생성 시 케이크의 디자인은 복잡한 데코를 사용하지 말고, 평면 그림 위주로 구성한다."
)

# =============================================================
#  API 키 로딩 / 클라이언트 초기화
# =============================================================
def load_api_key() -> str | None:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    if os.path.isfile(API_KEY_FILE):
        with open(API_KEY_FILE, "r", encoding="utf-8") as file:
            file_key = file.read().strip()
            if file_key:
                return file_key
    return None

OPENAI_API_KEY = load_api_key()
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY 환경 변수 또는 openai_key.txt 파일을 통해 API 키를 제공해주세요.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
HAS_RESPONSES_API = bool(client and getattr(getattr(client, "responses", None), "create", None))
HAS_IMAGES_API = bool(client and getattr(getattr(client, "images", None), "generate", None))

DEFAULT_IMAGE_MODEL = "dall-e-3"
ALT_IMAGE_MODEL = "dall-e-3"

# =============================================================
#  유틸리티
# =============================================================
def calculate_price(order):
    base = MENU["sizes"].get(order.get('size'), 0)
    filling = MENU["fillings"].get(order.get('filling'), 0)
    custom_fee = MENU["base_custom"]
    extra_cost = 0

    if order.get('has_image'): extra_cost += MENU['extras']['image']
    if order.get('has_color'): extra_cost += MENU['extras']['color']
    obj_count = order.get('object_count', 0)
    extra_cost += (obj_count * MENU['extras']['object'])
    lettering = order.get('lettering', '')
    if len(lettering) >= 10: extra_cost += MENU['extras']['long_lettering']

    return base + filling + custom_fee + extra_cost


def encode_image(uploaded_file) -> str:
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)
    return base64.b64encode(uploaded_file.read()).decode("utf-8")


def build_prompt(user_prompt: str, system_prompt: str) -> str:
    return f"{system_prompt}\n\n사용자 요청:\n{user_prompt}"

def extract_output_text(response):
    """
    Responses API output 에서 텍스트를 추출하는 안전한 헬퍼 함수
    dict / object 타입 모두 안전하게 처리
    """
    if not hasattr(response, "output") or response.output is None:
        # fallback (일부 모델은 output_text 를 직접 제공)
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        return ""

    final_text = ""

    # output이 없을 수도 있음
    output = getattr(response, "output", None)
    if not output:
        # 일부 모델은 output_text로만 제공함
        if hasattr(response, "output_text"):
            return response.output_text or ""
        return ""

    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue

        for c in content:

            # 1) 객체 타입(content.item이 object 형태) 처리
            if hasattr(c, "type") and c.type == "output_text":
                if hasattr(c, "text"):
                    final_text += (c.text or "")

            # 2) dict 타입 처리
            elif isinstance(c, dict):
                if c.get("type") == "output_text":
                    final_text += c.get("text", "")

            # 3) ResponseReasoningItem처럼 dict도 아니고 type/text가 없는 경우는 무시
            else:
                continue

    return final_text


# =============================================================
#  GPT 분석: 채팅에서 사용자의 의도(디자인 요소) 추출
# =============================================================
def analyze_intent_with_gpt(user_text, current_order, chat_history):
    if not client:
        return current_order, "🚨 OpenAI 클라이언트가 초기화되지 않았습니다. API 키를 확인해주세요."

    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
    history_str = json.dumps(recent_history, ensure_ascii=False)

    system_prompt = f"""
너는 '주문제작 케이크' 상담원이야. 고객의 말에서 디자인 요소를 추출해.
[현재 주문] {json.dumps(current_order, ensure_ascii=False)}
[대화 기록] {history_str}

[분석 규칙]
1. 'design_desc': 디자인 묘사 요약.
2. 'lettering': 레터링 문구.
3. 'has_color' (Boolean): 색상 변경 시 true.
4. 'object_count' (Integer): 추가 장식물 개수.

[응답 포맷 (JSON)]
{{ 
    "updated_order": {{ "design_desc": "...", "lettering": "...", "has_color": true/false, "object_count": 0 }}, 
    "response_message": "..." 
}}
"""

    try:
        if HAS_RESPONSES_API:
            content = [
                {"type": "input_text", "text": system_prompt},
                {"type": "input_text", "text": user_text},
            ]

            response = client.responses.create(
                model="gpt-5-nano",
                input=[{
                    "role": "user",
                    "content": content
                }],
            )

            content_str = extract_output_text(response).strip()

            content_str = content_str.strip()
        else:
            return current_order, "지원되는 챗 API가 없습니다."

        if "```" in content_str:
            m = re.search(r"\{.*\}", content_str, re.DOTALL)
            if m:
                content_str = m.group()
        try:
            result = json.loads(content_str)
        except Exception:
            m = re.search(r"\{.*\}", content_str, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                except Exception as e:
                    return current_order, f"파싱 오류: 응답을 JSON으로 변환할 수 없습니다. (원문: {content_str[:300]})"
            else:
                return current_order, f"응답을 이해할 수 없습니다: {content_str[:300]}"

        new_order = current_order.copy()
        if result.get('updated_order'):
            new_order.update(result['updated_order'])
        response_message = result.get('response_message', '요청을 반영했습니다.')
        return new_order, response_message

    except Exception as e:
        return current_order, f"오류 발생: {str(e)}"

# =============================================================
#  디자인 브리핑 & 이미지 생성
# =============================================================
def request_design_brief(user_prompt: str, system_prompt: str, image_b64: str | None = None, filling: str = "") -> str:
    if not client:
        return "OpenAI 클라이언트가 초기화되지 않았습니다."

    filling_context = ""
    if filling:
        if filling == "초코":
            filling_context = "\n\n중요: 이 케이크는 초코 케이크입니다. 전체적으로 다크하고 고급스러운 초콜릿 분위기로, 너무 화려하지 않고 차분하게 디자인해야 합니다."
        elif filling == "생크림":
            filling_context = "\n\n중요: 이 케이크는 생크림 케이크입니다. 전체적으로 밝고 깔끔한 생크림 분위기로, 파스텔 톤의 심플한 디자인을 사용하세요."
        elif filling == "레드벨벳":
            filling_context = "\n\n중요: 이 케이크는 레드벨벳 케이크입니다. 레드와 화이트의 조화를 살리되, 과도한 장식 없이 고급스러운 느낌을 유지하세요."
        elif filling == "티라미수":
            filling_context = "\n\n중요: 이 케이크는 티라미수 케이크입니다. 카카오와 크림의 조화를 살린, 차분하고 성숙한 분위기의 심플한 디자인이어야 합니다."
    enhanced_prompt = build_prompt(user_prompt + filling_context, system_prompt)

    if HAS_RESPONSES_API:
        content = [{"type": "input_text", "text": enhanced_prompt}]
        if image_b64:
            content.append({
                "type": "input_image",
                "data": {
                    "image": image_b64
                }
            })




        response = client.responses.create(
                model="gpt-5-nano",
                input=[{
                    "role": "user",
                    "content": content
                }],
        )
        primary_text = extract_output_text(response)
        return primary_text or "결과를 읽어오지 못했습니다."



    return "지원되는 챗 API가 없습니다."


# ✅ 현실적인, 과하지 않은 1단 케이크 이미지를 강제하는 이미지 프롬프트
def build_image_prompt(user_prompt: str, design_brief: str, filling: str = "") -> str:
    filling_mood = ""
    if filling == "초코":
        filling_mood = "다크하고 고급스러운 초콜릿 분위기이지만, 장식은 과하지 않고 차분한 느낌의 심플한 디자인."
    elif filling == "생크림":
        filling_mood = "밝고 깔끔한 생크림 분위기. 파스텔 톤 위주의 미니멀한 디자인."
    elif filling == "레드벨벳":
        filling_mood = "우아하고 고급스러운 레드벨벳 분위기. 레드와 화이트의 단순한 조합."
    elif filling == "티라미수":
        filling_mood = "티라미수 특유의 카카오와 크림 조화. 브라운/크림 톤의 차분한 디자인."
    filling_context = f"\n케이크 맛: {filling}\n{filling_mood}\n" if filling_mood else f"\n케이크 맛: {filling}\n"

    return f"""
You are a pâtisserie for custom cake.

CONSTRAINTS (MUST FOLLOW):
- Render only a REALISTIC, physically feasible 1-tier (single-layer) cake.
- Never produce multi-tier or floating/structurally impossible cakes.
- The cake must look like a real custom cake you could order at a small local Korean bakery or home bakery, NOT a luxury wedding cake or fantasy cake.
- Overall style should be simple, minimal, and easy to make in a real kitchen.
- Limit the color palette to at most 2–3 main colors.
- Do NOT use tall 3D toppers, big figurines, or complex sculptures. Decorations must stay low-profile: cream piping, small fruits, small chocolate pieces, simple flat drawings on the top surface, etc.
- Use only real bakery materials (buttercream, fresh cream, fruits, chocolate, simple sugar flowers, edible gold flakes, etc.).
- No text overlays, no watermarks, no logos in the image itself.
- Showcase the cake as a real product photo in a clean studio setting, shallow depth of field, close-up.
- The cake must be easy and realistic for a real baker to reproduce.

{filling_context}
User request (Korean):
{user_prompt}

Design brief (Korean):
{design_brief}

Output target:
A realistic product hero image of a single-tier, simple, minimal custom cake that a real bakery can easily make.
""".strip()


def request_design_image(prompt: str, model: str = DEFAULT_IMAGE_MODEL) -> bytes | None:
    if not client:
        return None

    if HAS_IMAGES_API:
        kwargs = {
            "model": model,
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "high",
            "response_format": "b64_json",
        }

        try:
            response = client.images.generate(**kwargs)
        except BadRequestError as err:
            error_str = str(err)
            if "response_format" in error_str:
                kwargs.pop("response_format", None)
            if "quality" in error_str or "Invalid value" in error_str:
                kwargs.pop("quality", None)
            response = client.images.generate(**kwargs)
        except PermissionDeniedError:
            if model == DEFAULT_IMAGE_MODEL:
                return request_design_image(prompt, ALT_IMAGE_MODEL)
            raise

        image_b64 = response.data[0].b64_json
        return base64.b64decode(image_b64)

    return None

# =============================================================
#  UI helpers
# =============================================================

def render_sidebar_summary(order):
    price = calculate_price(order)

    design_txt = order.get('design_desc', '-')
    if design_txt != '-' and len(design_txt) > 50:
        design_txt = design_txt[:50] + "..."
    lettering_txt = order.get('lettering', '-')

    extras_html = ""
    if order.get('has_image'): extras_html += f"<li>📸 사진 추가 <small>(+10,000)</small></li>"
    if order.get('has_color'): extras_html += f"<li>🎨 색상 변경 <small>(+5,000)</small></li>"
    cnt = order.get('object_count', 0)
    if cnt > 0: extras_html += f"<li>🧸 오브제 {cnt}개 <small>(+{cnt*2000:,})</small></li>"
    if len(lettering_txt) >= 10: extras_html += f"<li>✒️ 긴 레터링 <small>(+3,000)</small></li>"

    if extras_html:
        extras_html = f"<hr style='margin:5px 0;'><ul style='padding-left: 20px; margin: 0; font-size: 13px; color: #555;'>{extras_html}</ul>"

    html_code = f"""
    <div style="background-color: #FFF0F5; border: 2px solid #FF4081; border-radius: 12px; padding: 15px; margin-bottom: 20px; color: #000000;">
        <h4 style="margin:0 0 10px 0; color:#FF4081; border-bottom:1px solid #FF80AB; padding-bottom:5px; font-weight:bold;">🧾 실시간 주문서</h4>
        <div style="font-size:14px; line-height:1.6; color:#333;">
            👤 <b>{order.get('name','-')}</b> 님<br>
            📅 {order.get('pickupDate','-')}<br>⏰ {order.get('pickupTime','-')}<br>
            <hr style="margin: 8px 0; border-top: 1px dashed #FF4081;">
            🎂 <b>{order.get('size','-')}</b><br>🍰 <b>{order.get('filling','-')}</b><br>
            <div style="background-color: white; padding: 10px; border-radius: 8px; margin-top: 10px; border: 1px solid #FFCDD2;">
                🎨 <b>디자인:</b> {design_txt}<br>
                ✏️ <b>레터링:</b> {lettering_txt}
                {extras_html}
            </div>
        </div>
        <div style="margin-top: 10px; text-align: right; font-size: 20px; font-weight: bold; color: #D32F2F;">{price:,}원</div>
    </div>
    """
    clean_html = html_code.replace("\n", "")
    st.sidebar.markdown(clean_html, unsafe_allow_html=True)

@st.dialog("🧾 최종 견적서 확인")
def show_final_confirmation(order, image_data):
    st.markdown("### 📋 주문 내역")
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.write(f"👤 **{order['name']}** 님"); st.write(f"📞 {order['phone']}")
    with c2: st.write(f"📅 **{order['pickupDate']}**"); st.write(f"⏰ **{order['pickupTime']}**")

    st.info(f"🎂 **{order['size']}** / **{order['filling']}**")
    st.success(f"🎨 디자인: {order.get('design_desc', '-')}\n\n✏️ 레터링: {order.get('lettering', '-')}")

    st.markdown("#### 💰 상세 견적")
    details = []
    if order.get('has_image'): details.append("사진 추가 (+10,000)")
    if order.get('has_color'): details.append("색상 변경 (+5,000)")
    if order.get('object_count', 0) > 0: details.append(f"오브제 {order['object_count']}개 (+{order['object_count']*2000:,})")
    if len(order.get('lettering', '')) >= 10: details.append("긴 레터링 (+3,000)")

    if details:
        for d in details: st.caption(f"- {d}")
    else:
        st.caption("- 기본 주문제작비 포함")

    if 'generated_design_image' in st.session_state and st.session_state.generated_design_image:
        st.markdown("#### 🎨 생성된 케이크 시안")
        st.image(st.session_state.generated_design_image, caption="AI 생성 케이크 시안", use_container_width=True)
    elif image_data:
        st.markdown("#### 📸 참고 디자인")
        st.image(image_data, caption="참고 디자인", use_container_width=True)
    st.divider()
    st.markdown(f"### 💰 총 결제금액: :red[{order['price']:,}원]")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", use_container_width=True): st.rerun()
    with c2:
        if st.button("전송 🚀", type="primary", use_container_width=True): st.session_state.step = 'SENT'; st.rerun()

# =============================================================
#  메인
# =============================================================
def main():
    st.markdown("""
    <style>
        .stApp, section[data-testid="stSidebar"] { background-color: #FFFFFF !important; color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("통합 주문제작 케이크 스튜디오")

    # session init
    if 'step' not in st.session_state: st.session_state.step = 'FORM'
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'order' not in st.session_state: st.session_state.order = {}
    if 'uploaded_img' not in st.session_state: st.session_state.uploaded_img = None

    # 답변/시안 생성 로직 실행 (UI는 블러 없이 말풍선만 사용)
    if st.session_state.get('process_on_next'):
        st.session_state.process_on_next = False
        prompt = st.session_state.get('pending_prompt', '')
        placeholder_idx = st.session_state.get('pending_placeholder_idx', None)
        try:
            clean_hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m.get('content')]
            new_order, ai_res = analyze_intent_with_gpt(prompt, st.session_state.order, clean_hist)
            st.session_state.order = new_order
            st.session_state.order['price'] = calculate_price(new_order)

            if st.session_state.get('auto_generate_design', False):
                img_b64 = None
                if st.session_state.uploaded_img:
                    img_b64 = encode_image(st.session_state.uploaded_img)

                filling = st.session_state.order.get('filling', '')
                design_context = st.session_state.order.get('design_desc', '-')
                lettering_context = st.session_state.order.get('lettering', '-')
                combined_prompt = f"사용자 요청: {prompt}\n\n(현재 반영된 디자인) 디자인: {design_context}\n레터링: {lettering_context}"

                design_brief = request_design_brief(
                    user_prompt=combined_prompt,
                    system_prompt=DEFAULT_DESIGN_SYSTEM_PROMPT,
                    image_b64=img_b64,
                    filling=filling
                )

                img_prompt = build_image_prompt(combined_prompt, design_brief, filling)
                img_bytes = request_design_image(img_prompt)

                if img_bytes:
                    st.session_state.generated_design_image = img_bytes
                    st.session_state.order['design_desc'] = design_brief
                    st.session_state.order['price'] = calculate_price(st.session_state.order)
                    final_msg = (
                        "✅ 시안이 생성되었습니다!\n\n"
                        "디자인 제안:\n"
                        f"{design_brief}\n\n"
                        "생성된 시안은 사이드바와 최종 견적서에서 확인할 수 있습니다."
                    )
                else:
                    final_msg = "시안 이미지 생성에 실패했습니다. (권한/모델 문제일 수 있음)\n\nAI 응답: " + ai_res
            else:
                final_msg = ai_res

            if placeholder_idx is not None and 0 <= placeholder_idx < len(st.session_state.messages):
                st.session_state.messages[placeholder_idx]['content'] = final_msg
            else:
                st.session_state.messages.append({"role": "assistant", "content": final_msg})

        except Exception as e:
            err_msg = f"처리 중 오류가 발생했습니다: {e}"
            if placeholder_idx is not None and 0 <= placeholder_idx < len(st.session_state.messages):
                st.session_state.messages[placeholder_idx]['content'] = err_msg
            else:
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

        st.session_state.pop('pending_prompt', None)
        st.session_state.pop('pending_placeholder_idx', None)
        st.rerun()

    # Sidebar
    with st.sidebar:
        if not OPENAI_API_KEY:
            st.error("API Key Missing")
        if st.session_state.step == 'CHAT':
            render_sidebar_summary(st.session_state.order)

        st.markdown("---")
        st.markdown("### ✅ 주문 확정")
        if st.button("최종 견적서 보기", type="primary", use_container_width=True):
            if st.session_state.step == 'CHAT':
                design_image = st.session_state.get('generated_design_image') or st.session_state.uploaded_img
                show_final_confirmation(st.session_state.order, design_image)
            else:
                st.warning("주문서를 먼저 작성해주세요!")
        st.markdown("---")

        st.subheader("🖼️ 참고 사진")
        if st.session_state.step == 'CHAT':
            uploaded = st.file_uploader("참고할 사진을 업로드 하세요!", type=["png", "jpg", "jpeg"])

            if uploaded and st.session_state.get('last_img') != getattr(uploaded, "name", None):
                st.session_state.last_img = getattr(uploaded, "name", None)
                st.session_state.uploaded_img = uploaded
                st.session_state.order['has_image'] = True
                st.session_state.order['price'] = calculate_price(st.session_state.order)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "참고 사진이 추가되었습니다. (+10,000원)"
                })
                st.rerun()

            if st.session_state.uploaded_img is not None:
                st.image(st.session_state.uploaded_img, caption="업로드된 디자인", use_container_width=True)
                if st.button("참고 사진 제거", key="remove_uploaded_img"):
                    st.session_state.uploaded_img = None
                    st.session_state.last_img = None
                    st.session_state.order['has_image'] = False
                    st.session_state.order['price'] = calculate_price(st.session_state.order)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "참고 사진이 제거되었습니다."
                    })
                    st.rerun()
        else:
            st.info("상담이 시작되면 사진을 올릴 수 있어요.")

        st.markdown("---")
        st.markdown("### 🎨 시안 자동 생성")
        st.caption("토글을 켜면 채팅 입력 시 자동으로 시안이 생성됩니다.")
        if 'auto_generate_design' not in st.session_state:
            st.session_state.auto_generate_design = False
        auto_generate = st.toggle(
            "시안 생성",
            value=st.session_state.auto_generate_design,
            key="auto_generate_toggle"
        )
        st.session_state.auto_generate_design = auto_generate

        # ✅ 요구사항: 시안 자동 생성 아래에 생성된 시안 미리보기
        if st.session_state.get('generated_design_image'):
            st.markdown("#### ✨ 생성된 시안 미리보기")
            st.image(
                st.session_state.generated_design_image,
                caption="AI 생성 케이크 시안",
                use_container_width=True
            )

    # FORM step
    if st.session_state.step == 'FORM':
        st.markdown("##### 👇 필수 정보를 입력해주세요")
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("주문자 성함")
            with c2:
                phone = st.text_input("연락처")
            c3, c4 = st.columns(2)
            with c3:
                size = st.selectbox("사이즈", list(MENU["sizes"].keys()))
            with c4:
                fill = st.selectbox("맛", list(MENU["fillings"].keys()))
            c5, c6 = st.columns(2)
            with c5:
                date = st.selectbox("픽업 날짜", list(SCHEDULE.keys()))
            with c6:
                times = SCHEDULE.get(date, [])
                time_sel = st.selectbox("픽업 시간", times) if times else None

            if st.button("상담 시작하기 💬", type="primary", use_container_width=True):
                if not name or not phone or not time_sel:
                    st.error("⚠️ 모든 정보를 입력해주세요!")
                else:
                    st.session_state.order = {
                        'name': name,
                        'phone': phone,
                        'size': size,
                        'filling': fill,
                        'decoration': '주문제작',
                        'has_image': False,
                        'has_color': False,
                        'object_count': 0,
                        'pickupDate': date,
                        'pickupTime': time_sel,
                        'design_desc': '-',
                        'lettering': '-'
                    }
                    st.session_state.order['price'] = calculate_price(st.session_state.order)
                    welcome_msg = (
                        f"안녕하세요 {name}님! 👋 왼쪽 주문서 보이시죠?\n\n"
                        "원하는 케이크의 디자인과 레터링 문구를 적어주세요.\n\n"
                        "케이크 디자인 : \n"
                        "레터링 : \n"
                        "(왼쪽에 사진에 초안을 업로드해주면 최고~!)\n\n"
                        "케이크 디자인은 최대한 상세하게 적어주세요 😊"
                    )
                    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                    st.session_state.step = 'CHAT'
                    st.rerun()

    # CHAT step
    elif st.session_state.step == 'CHAT':
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="background-color: #E3F2FD; color: black; padding: 12px; border-radius: 15px 15px 0 15px; border: 1px solid #BBDEFB; max-width: 70%;">{msg['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="background-color: #F5F5F5; color: black; padding: 12px; border-radius: 15px 15px 15px 0; border: 1px solid #E0E0E0; max-width: 70%;">{msg['content'].replace(chr(10), '<br>')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Chat input
        if prompt := st.chat_input("메시지 입력..."):
            # 1) 사용자가 보낸 채팅을 바로 말풍선에 표시
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 2) AI 로딩 상태를 '답변/시안 생성 중' 문구로만 표시 (화면 블러 없음)
            placeholder_idx = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                # ✅ 요구사항 1: '답변/시안 생성중' 문구 (블러/모달 없이 말풍선으로만 표시)
                "content": "⏳ 답변/시안 생성 중입니다. 잠시만 기다려 주세요!"
            })

            # 3) 실제 처리는 다음 run에서 수행
            st.session_state.pending_prompt = prompt
            st.session_state.pending_placeholder_idx = placeholder_idx
            st.session_state.process_on_next = True
            st.rerun()

    # SENT step
    elif st.session_state.step == 'SENT':
        st.balloons()
        st.success("전송 완료!")
        if st.button("처음으로"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
