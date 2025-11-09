# Import Libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ---------------- CUSTOM PAGE DESIGN ---------------- #
st.set_page_config(page_title="UMKM Business Helper Bot", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #eef0f3;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
        padding: 8px;
    }
    .stChatMessage {
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ---------------- #
st.title("UMKM Business Helper Bot")
st.write("Asisten pemasaran digital untuk membantu UMKM membuat konten pemasaran yang efektif.")


# ---------------- SIDEBAR SETTINGS ---------------- #
with st.sidebar:
    st.header("Settings")

    google_api_key = st.text_input("Google AI API Key", type="password")

    st.markdown("---")
    st.subheader("Preferensi Marketing")
    default_style = st.selectbox("Gaya Bahasa", ["Ramah & Persuasif", "Formal", "Santai / Gaul"], index=0)
    default_platform = st.selectbox("Platform Promosi", ["Instagram", "TikTok", "Marketplace"], index=0)

    reset_button = st.button("Reset", help="Reset percakapan dan memory produk")


if not google_api_key:
    st.info("Masukkan Google API key untuk mulai chatting.")
    st.stop()


# ---------------- FORM INPUT PRODUCT MEMORY ---------------- #
st.subheader("Detail Produk UMKM")

with st.form("product_info_form"):
    product_name = st.text_input("Nama Produk", st.session_state.get("product_name", ""))
    product_category = st.text_input("Kategori Produk", st.session_state.get("product_category", ""))
    product_target = st.text_input("Target Audience", st.session_state.get("product_target", ""))
    product_price = st.text_input("Kisaran Harga (opsional)", st.session_state.get("product_price", ""))

    save_product_info = st.form_submit_button("Simpan Informasi Produk")

if save_product_info:
    st.session_state.product_name = product_name
    st.session_state.product_category = product_category
    st.session_state.product_target = product_target
    st.session_state.product_price = product_price
    st.success("Informasi produk berhasil disimpan.")


# ---------------- TOOL FUNCTIONS (callables with name attr) ---------------- #
def price_estimator_tool(category: str) -> str:
    """Return an estimated price range for a product category."""
    # normalize category to simple key
    if not category:
        return "Kategori tidak disebutkan. Mohon sebutkan kategori produk (mis: Makanan, Fashion, Kerajinan)."
    key = category.strip().title()
    price_map = {
        "Makanan": "Rp10.000 - Rp40.000",
        "Minuman": "Rp8.000 - Rp30.000",
        "Fashion": "Rp50.000 - Rp200.000",
        "Kerajinan": "Rp30.000 - Rp150.000",
        "Jasa": "Rp50.000 - Rp250.000",
        "Lainnya": "Harga bervariasi tergantung pasar lokal"
    }
    return price_map.get(key, "Harga belum tersedia untuk kategori tersebut.")

# attach metadata attributes the agent/adapter may check
price_estimator_tool.name = "PriceEstimator"
price_estimator_tool.description = "Memperkirakan rentang harga berdasarkan kategori produk UMKM."


def hashtag_generator_tool(category: str) -> str:
    """Return suggested hashtags for a given category."""
    if not category:
        return "#umkm #jualanonline"
    key = category.strip().title()
    tag_map = {
        "Makanan": "#kuliner #jajanmurah #makananlezat #cemilan",
        "Fashion": "#ootd #fashionlokal #brandlokal #gayalokal",
        "Kerajinan": "#produkhandmade #umkmlokal #karyaanakbangsa",
        "Jasa": "#layananprofesional #jasaindonesia #bisnisjasa",
        "Lainnya": "#umkm #jualanonline #bisnisrumahan"
    }
    return tag_map.get(key, "#umkmindonesia")

hashtag_generator_tool.name = "HashtagGenerator"
hashtag_generator_tool.description = "Membuat saran hashtag berdasarkan kategori produk."


# ---------------- HELPER PROMPT ---------------- #
def build_umkm_user_prompt(user_question: str, style: str, platform: str):
    template = f"""
    Kamu adalah UMKM Business Helper Bot.

    Tugas kamu:
    - Membuat deskripsi produk singkat dan menarik
    - Memberikan ide caption promosi untuk platform {platform}
    - Memberikan strategi pemasaran sederhana
    - Memberikan saran rentang harga jika relevan

    Format jawaban:

    1. Deskripsi Produk
       - 2-3 kalimat promosi

    2. Caption Promosi ({platform})
       - Caption 1
       - Caption 2
       - Caption 3

    3. Strategi Promosi Singkat
       - Strategi 1
       - Strategi 2
       - Strategi 3

    4. Saran Harga (opsional)
       - Range harga wajar sesuai kategori produk

    Gunakan gaya bahasa: {style}

    Permintaan pengguna:
    "{user_question}"
    """
    return template


# ---------------- INIT AGENT ---------------- #
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.85,
            top_p=0.9
        )

        system_prompt = """
        You are UMKM Business Helper Bot, an AI assistant for Indonesian SMEs.
        Always respond in Indonesian. If needed, ask for clarifying details.
        Provide structured and helpful marketing suggestions.
        """

        # pass the callable functions as tools (they have .name and .description attributes)
        tools = [price_estimator_tool, hashtag_generator_tool]

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt
        )

        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)

    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()


# ---------------- CHAT HISTORY ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    # clear only keys we used
    for k in ["messages", "agent", "product_name", "product_category", "product_target", "product_price", "_last_key"]:
        if k in st.session_state:
            st.session_state.pop(k, None)
    st.rerun()


# ---------------- DISPLAY CHAT HISTORY ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- USER INPUT ---------------- #
prompt = st.chat_input("Masukkan kebutuhan pemasaran produk Anda...")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        formatted_prompt = build_umkm_user_prompt(prompt, default_style, default_platform)

        messages = []

        # Inject Product Memory as System Instruction
        product_memory = ""
        if "product_name" in st.session_state:
            product_memory += f"Nama Produk: {st.session_state.product_name}\n"
        if "product_category" in st.session_state:
            product_memory += f"Kategori: {st.session_state.product_category}\n"
        if "product_target" in st.session_state:
            product_memory += f"Target Audience: {st.session_state.product_target}\n"
        if "product_price" in st.session_state:
            product_memory += f"Harga Ideal: {st.session_state.product_price}\n"

        if product_memory:
            messages.append(SystemMessage(content="Informasi Produk:\n" + product_memory))

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=formatted_prompt))

        # Invoke agent
        response = st.session_state.agent.invoke({"messages": messages})

        # extract assistant text robustly (langgraph responses can vary)
        answer = ""
        if isinstance(response, dict) and "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        elif hasattr(response, "content"):
            # fallback if agent returns a single message object
            answer = response.content
        else:
            answer = str(response)

    except Exception as e:
        answer = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
