import streamlit as st
import os
import time
import random
import json
import re
from openai import OpenAI

# =============================================================
#  API 키 로딩 / 클라이언트 초기화
# =============================================================
API_KEY_FILE = "openai_key.txt"

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

# --- 1. 데이터 ---
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

# --- 2. 로직 ---
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

def analyze_intent_with_gpt(user_text, current_order, chat_history):
    if "sk-" not in OPENAI_API_KEY:
        return current_order, "🚨 API 키가 설정되지 않았습니다!"

    client = OpenAI(api_key=OPENAI_API_KEY)
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
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        if "```" in content: content = re.search(r"\{.*\}", content, re.DOTALL).group()
        result = json.loads(content)
        new_order = current_order.copy()
        if result.get('updated_order'): new_order.update(result['updated_order'])
        return new_order, result['response_message']
    except Exception as e:
        return current_order, f"오류 발생: {str(e)}"

# --- 3. UI 컴포넌트 (깨짐 현상 완벽 해결) ---
def render_sidebar_summary(order):
    price = calculate_price(order)
    
    design_txt = order.get('design_desc', '-')
    lettering_txt = order.get('lettering', '-')
    
    extras_html = ""
    if order.get('has_image'): extras_html += f"<li>📸 사진 추가 <small>(+10,000)</small></li>"
    if order.get('has_color'): extras_html += f"<li>🎨 색상 변경 <small>(+5,000)</small></li>"
    cnt = order.get('object_count', 0)
    if cnt > 0: extras_html += f"<li>🧸 오브제 {cnt}개 <small>(+{cnt*2000:,})</small></li>"
    if len(lettering_txt) >= 10: extras_html += f"<li>✒️ 긴 레터링 <small>(+3,000)</small></li>"

    if extras_html:
        extras_html = f"<hr style='margin:5px 0;'><ul style='padding-left: 20px; margin: 0; font-size: 13px; color: #555;'>{extras_html}</ul>"

    # 🔥 [수정] HTML을 한 줄로 연결하여 들여쓰기 버그 원천 차단 🔥
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
    # 줄바꿈 문자를 공백으로 치환하여 마크다운 파서 오류 방지
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

    if image_data: st.image(image_data, caption="참고 디자인", use_column_width=True)
    st.divider()
    st.markdown(f"### 💰 총 결제금액: :red[{order['price']:,}원]")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", use_container_width=True): st.rerun()
    with c2:
        if st.button("전송 🚀", type="primary", use_container_width=True): st.session_state.step = 'SENT'; st.rerun()

# --- 4. 메인 실행 ---
def main():
    st.set_page_config(page_title="주문제작 케이크 상담하기", layout="wide")

    # CSS 유지
    st.markdown("""
    <style>
        .stApp, section[data-testid="stSidebar"] { background-color: #FFFFFF !important; color: #000000 !important; }
        p, h1, h2, h3, h4, h5, h6, li, label, span, div, small { color: #000000 !important; }
        section[data-testid="stSidebar"] { border-right: 2px solid #E0E0E0; background-color: #FAFAFA !important; }
        div.stButton > button[kind="secondary"] { background-color: #FFFFFF !important; color: #000000 !important; border: 1px solid #999 !important; }
        div.stButton > button[kind="primary"] p { color: #FFFFFF !important; }
        .stTextInput input { background-color: #FFFFFF !important; color: #000000 !important; border: 1px solid #CCC !important; }
        [data-testid="stFileUploader"] section { background-color: #FFFFFF !important; border: 1px dashed #999 !important; }
        .stSelectbox div[data-baseweb="select"] > div, ul[data-baseweb="menu"], li[data-baseweb="option"] {
            background-color: #FFFFFF !important; color: #000000 !important;
        }
        div[data-testid="stChatMessage"]:nth-child(odd) { flex-direction: row-reverse; }
        div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] { background-color: #E3F2FD !important; color: black !important; border: 1px solid #BBDEFB; }
        div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] { background-color: #F1F3F4 !important; color: black !important; border: 1px solid #DADCE0; }
        .stChatInput { border-top: 2px solid #FF4081 !important; padding-top: 15px !important; background-color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("주문제작 케이크 상담하기")
    
    if 'step' not in st.session_state: st.session_state.step = 'FORM'
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'order' not in st.session_state: st.session_state.order = {}
    if 'uploaded_img' not in st.session_state: st.session_state.uploaded_img = None

    with st.sidebar:
        if "sk-" not in OPENAI_API_KEY: st.error("API Key Missing")
        if st.session_state.step == 'CHAT': render_sidebar_summary(st.session_state.order)
        st.markdown("---")
        st.markdown("### ✅ 주문 확정")
        if st.button("최종 견적서 보기", type="primary", use_container_width=True):
            if st.session_state.step == 'CHAT': show_final_confirmation(st.session_state.order, st.session_state.uploaded_img)
            else: st.warning("주문서를 먼저 작성해주세요!")
        st.markdown("---")
        st.subheader("🖼️ 디자인 사진")
        if st.session_state.step == 'CHAT':
            uploaded = st.file_uploader("참고할 사진이 있나요?", type=["png", "jpg"])
            if uploaded: st.image(uploaded, caption="업로드된 디자인", use_column_width=True)
            
            if uploaded and st.session_state.get('last_img') != uploaded.name:
                st.session_state.last_img = uploaded.name; st.session_state.uploaded_img = uploaded
                st.session_state.order['has_image'] = True 
                st.session_state.order['price'] = calculate_price(st.session_state.order)
                st.session_state.messages.append({"role": "assistant", "content": "참고 사진이 추가되었습니다. (+10,000원)"}); st.rerun()
            
            elif not uploaded and st.session_state.uploaded_img is not None:
                st.session_state.uploaded_img = None; st.session_state.last_img = None
                st.session_state.order['has_image'] = False 
                st.session_state.order['price'] = calculate_price(st.session_state.order)
                st.session_state.messages.append({"role": "assistant", "content": "참고 사진이 제거되었습니다."}); st.rerun()
        else: st.info("상담이 시작되면 사진을 올릴 수 있어요.")

    if st.session_state.step == 'FORM':
        st.markdown("##### 👇 필수 정보를 입력해주세요")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1: name = st.text_input("주문자 성함")
            with c2: phone = st.text_input("연락처")
            c3, c4 = st.columns(2)
            with c3: size = st.selectbox("사이즈", list(MENU["sizes"].keys()))
            with c4: fill = st.selectbox("맛", list(MENU["fillings"].keys()))
            c5, c6 = st.columns(2)
            with c5: date = st.selectbox("픽업 날짜", list(SCHEDULE.keys()))
            with c6:
                times = SCHEDULE.get(date, [])
                time_sel = st.selectbox("픽업 시간", times) if times else None
            
            if st.button("상담 시작하기 💬", type="primary", use_container_width=True):
                if not name or not phone or not time_sel: st.error("⚠️ 모든 정보를 입력해주세요!")
                else:
                    st.session_state.order = {
                        'name':name, 'phone':phone, 'size':size, 'filling':fill, 
                        'decoration':'주문제작', 
                        'has_image': False, 'has_color': False, 'object_count': 0, 
                        'pickupDate':date, 'pickupTime':time_sel, 
                        'design_desc': '-', 'lettering': '-'
                    }
                    st.session_state.order['price'] = calculate_price(st.session_state.order)
                    welcome_msg = f"""안녕하세요 **{name}**님! 👋 왼쪽 주문서 보이시죠?\n\n원하는 케이크의 디자인과 레터링 문구를 적어주세요.\n\n케이크 디자인 : \n레터링 : \n(왼쪽에 사진에 초안을 업로드해주면 최고~!)\n\n케이크 디자인은 최대한 상세하게 적어주세요 😊"""
                    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                    st.session_state.step = 'CHAT'; st.rerun()

    elif st.session_state.step == 'CHAT':
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    st.markdown(f"""<div style="display: flex; justify-content: flex-end; margin-bottom: 10px;"><div style="background-color: #E3F2FD; color: black; padding: 12px; border-radius: 15px 15px 0 15px; border: 1px solid #BBDEFB; max-width: 70%;">{msg['content']}</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="display: flex; justify-content: flex-start; margin-bottom: 10px;"><div style="background-color: #F5F5F5; color: black; padding: 12px; border-radius: 15px 15px 15px 0; border: 1px solid #E0E0E0; max-width: 70%;">{msg['content'].replace(chr(10), '<br>')}</div></div>""", unsafe_allow_html=True)
        
        if prompt := st.chat_input("메시지 입력..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("AI가 입력 중..."):
                clean_hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                new_order, ai_res = analyze_intent_with_gpt(prompt, st.session_state.order, clean_hist)
            st.session_state.order = new_order
            st.session_state.order['price'] = calculate_price(new_order)
            st.session_state.messages.append({"role": "assistant", "content": ai_res}); st.rerun()

    elif st.session_state.step == 'SENT':
        st.balloons(); st.success("전송 완료!"); 
        if st.button("처음으로"): st.session_state.clear(); st.rerun()

if __name__ == "__main__":
    main()