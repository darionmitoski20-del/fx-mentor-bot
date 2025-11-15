import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from aiohttp import web
from dotenv import load_dotenv
from openai import OpenAI

# ============================
# CONFIG
# ============================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not BOT_TOKEN or not OPENAI_API_KEY or not TWELVEDATA_API_KEY:
    raise RuntimeError("BOT_TOKEN, OPENAI_API_KEY или TWELVEDATA_API_KEY недостигаат.")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

PAIR_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD",
    "AUDUSD": "AUD/USD",
    "USDJPY": "USD/JPY",
}

WATCH_INTERVAL_SECONDS = 60

# ============================
# DATA STRUCTURES
# ============================

@dataclass
class Zone:
    pair: str
    bias: str
    upper_zone: Optional[Tuple[float, float]] = None
    lower_zone: Optional[Tuple[float, float]] = None
    note: str = ""

@dataclass
class UserState:
    zones: Dict[str, Zone] = field(default_factory=dict)

USERS: Dict[int, UserState] = {}

def get_user_state(user_id: int) -> UserState:
    if user_id not in USERS:
        USERS[user_id] = UserState()
    return USERS[user_id]

# ============================
# HELPERS
# ============================

def parse_range(text: str) -> Optional[Tuple[float, float]]:
    m = re.match(r"\s*([0-9\.]+)\s*-\s*([0-9\.]+)\s*", text)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2))
    return (min(a, b), max(a, b))

def parse_plan(text: str) -> dict:
    def find(key):
        m = re.search(rf"{key}\s*:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    pair = (find("pair") or "").upper().replace("/", "")
    bias = (find("bias") or "").lower()
    upper_zone = parse_range(find("upper_zone") or "")
    lower_zone = parse_range(find("lower_zone") or "")
    rr = find("rr") or ""
    reason = find("reason") or ""

    return {
        "pair": pair,
        "bias": bias,
        "upper_zone": upper_zone,
        "lower_zone": lower_zone,
        "rr": rr,
        "reason": reason,
    }

def parse_check(text: str) -> dict:
    def find(key):
        m = re.search(rf"{key}\s*:\s*([A-Za-z0-9\.\-]+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    def ffloat(v):
        try:
            return float(v)
        except:
            return None

    pair = (find("pair") or "").upper().replace("/", "")
    direction = (find("direction") or "").lower()
    entry = ffloat(find("entry"))
    sl = ffloat(find("sl"))
    tp = ffloat(find("tp"))

    m_reason = re.search(r"reason\s*:\s*(.+)", text, re.IGNORECASE)
    reason = m_reason.group(1).strip() if m_reason else ""

    return {
        "pair": pair,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "reason": reason,
    }

def calc_rr(entry, sl, tp):
    if not entry or not sl or not tp:
        return None
    risk = abs(entry - sl)
    reward = abs(entry - tp)
    if risk == 0:
        return None
    return risk, reward, reward / risk

async def fetch_price(symbol: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=10) as s:
            r = await s.get(
                "https://api.twelvedata.com/price",
                params={"symbol": symbol, "apikey": TWELVEDATA_API_KEY},
            )
            data = r.json()
            if "price" in data:
                return float(data["price"])
    except:
        return None
    return None

# ============================
# OPENAI ANALYSIS
# ============================

async def ai_analyze_plan(plan, text):
    upper = plan["upper_zone"]
    lower = plan["lower_zone"]

    prompt = f"""
Ти си FX price action ментор. Корисникот е почетник.

Анализирај го планот:
- пар: {plan['pair']}
- bias: {plan['bias']}
- горна зона: {upper}
- долна зона: {lower}
- RR преференција: {plan['rr']}
- причина: {plan['reason']}

Објасни:
1) Дали bias има смисла
2) Како изгледаат зоните теоретски
3) Каде од ОКОЛУ би имало логични области за entry, SL, TP (без точни цени)
4) Што е добро и што е ризично
5) Сè да биде едукативно, НЕ финансиски совет.
"""

    r = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )

    return r.output[0].content[0].text

async def ai_check_setup(check, zone):
    rr = calc_rr(check["entry"], check["sl"], check["tp"])
    if rr:
        risk, reward, rr_val = rr
        rr_line = f"Ризик: {risk:.5f}, награда: {reward:.5f}, RR≈{rr_val:.2f}"
    else:
        rr_line = "Не може да се пресмета RR."

    prompt = f"""
FX ментор: провери го сетапот.

PAIR: {check['pair']}
Direction: {check['direction']}
Entry: {check['entry']}
SL: {check['sl']}
TP: {check['tp']}
{rr_line}

Зони од планот:
{zone}

Објасни:
- дали насоката има смисла со bias
- дали SL/TP се поставени логично
- дали RR е здрав
- на што да внимава
- едукативно, без сигнали
"""

    r = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )

    return r.output[0].content[0].text

async def ai_analyze_chart_image(image_url: str, caption: str) -> str:
    """
    Анализа на chart слика + текст од caption.
    """
    prompt = (
        "Ти си FX/crypto price action ментор.\n"
        "Корисникот ти праќа screenshot од чарт и кратко објаснување во caption.\n"
        "Твојата задача е:\n"
        "1) Да кажеш каков е приближно трендот (горе/долу/sideways) според чарот.\n"
        "2) Да опишеш важни зони (support/resistance, demand/supply), без да измислуваш точни цени.\n"
        "3) Да поврзеш со тоа што корисникот го пишал во caption (pair, TF, bias, план).\n"
        "4) Да предложиш потенцијално сценарио ОД ОКОЛУ (пример: ако bias е long, што би чекал: retest, break, "
        "confirmation на помал TF…), но без директни сигнали за влез.\n"
        "5) Да укажеш на ризици (fake break, слаба структура, нема јасен тренд итн.).\n"
        "6) Сè да биде едукативно, јасно и на македонски.\n\n"
        f"Caption од корисникот:\n{caption}\n\n"
        "Одговори во неколку јасни секции: Тренд, Зони, Идеи, Ризици.\n"
        "Не давај директни наредби за влез/излез, само објаснувај логика."
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    )

    return response.output[0].content[0].text


# ============================
# TELEGRAM COMMANDS
# ============================

@dp.message(Command("start"))
async def cmd_start(m: Message):
    await m.answer(
        "👋 Здраво! FX Mentor Bot е активен.\n\n"
        "Команди:\n"
        "/plan – постави зони\n"
        "/check – провери сетап\n"
        "/zones – активни зони\n"
        "/clear – избриши зони\n"
        "/help – помош\n\n"
        "Сè е едукативно, не е финансиски совет."
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "📘 Помош:\n\n"
        "**/plan**\n"
        "pair: EURUSD\n"
        "bias: short\n"
        "upper_zone: 1.0850-1.0870\n"
        "lower_zone: 1.0760-1.0780\n"
        "rr: 1:2\n"
        "reason: H4 downtrend\n\n"
        "**/check**\n"
        "pair: EURUSD\n"
        "direction: short\n"
        "entry: 1.0860\n"
        "sl: 1.0880\n"
        "tp: 1.0820",
        parse_mode="Markdown",
    )

@dp.message(Command("clear"))
async def cmd_clear(m: Message):
    USERS[m.from_user.id] = UserState()
    await m.answer("🧹 Зоните се избришани.")

@dp.message(Command("zones"))
async def cmd_zones(m: Message):
    st = get_user_state(m.from_user.id)
    if not st.zones:
        return await m.answer("Нема активни зони.")

    msg = "📍 Активни зони:\n"
    for p, z in st.zones.items():
        msg += f"\nPAIR: {p}\nBias: {z.bias}\nГорна: {z.upper_zone}\nДолна: {z.lower_zone}\n"
    await m.answer(msg)

@dp.message(Command("plan"))
async def cmd_plan(m: Message):
    text = m.text
    plan = parse_plan(text)

    if plan["pair"] not in PAIR_SYMBOLS:
        return await m.answer("❌ Непознат pair.")

    state = get_user_state(m.from_user.id)

    state.zones[plan["pair"]] = Zone(
        pair=plan["pair"],
        bias=plan["bias"],
        upper_zone=plan["upper_zone"],
        lower_zone=plan["lower_zone"],
        note=plan["reason"],
    )

    await m.answer("⏳ Анализирам...")
    analysis = await ai_analyze_plan(plan, text)
    await m.answer(analysis)

@dp.message(Command("check"))
async def cmd_check(m: Message):
    text = m.text
    data = parse_check(text)

    if data["pair"] not in PAIR_SYMBOLS:
        return await m.answer("❌ Непознат pair.")

    zones = get_user_state(m.from_user.id).zones.get(data["pair"])
    await m.answer("⏳ Проверувам сетап...")

    rez = await ai_check_setup(data, zones)
    await m.answer(rez)

@dp.message(Command("chart"))
async def cmd_chart(m: Message):
    """
    /chart команда за анализа на chart screenshot.

    КОРИСНИК:
    - испраќа слика од чарт
    - во caption пишува нешто како:
      /chart
      pair: BTCUSD
      tf: H1
      bias: long
      plan: гледам uptrend, можен retest...
    """
    if not m.photo:
        await m.answer(
            "За анализа на чарт, испрати screenshot како фотографија и во caption напиши, на пример:\n\n"
            "/chart\n"
            "pair: BTCUSD\n"
            "tf: H1\n"
            "bias: long\n"
            "plan: гледам uptrend, можен retest на зона"
        )
        return

    # земаме најголема верзија на сликата
    file_id = m.photo[-1].file_id
    file = await bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

    caption = m.caption or ""
    await m.answer("⏳ Го читам чартот, секундна...")

    try:
        analysis = await ai_analyze_chart_image(file_url, caption)
        await m.answer(analysis)
    except Exception as e:
        print("Chart analysis error:", e)
        await m.answer("Настана грешка при анализа на чартот. Провери дали сликата е јасна и пробај повторно.")


# ============================
# PRICE WATCHER (background)
# ============================

async def price_watcher():
    sent = set()
    while True:
        try:
            for uid, st in USERS.items():
                for pair, z in st.zones.items():
                    symbol = PAIR_SYMBOLS[pair]
                    price = await fetch_price(symbol)
                    if not price:
                        continue

                    if z.upper_zone:
                        low, high = z.upper_zone
                        if low <= price <= high and (uid, pair, "u") not in sent:
                            await bot.send_message(
                                uid,
                                f"📣 {pair} е во ГОРНАТА зона {low}-{high}.\nПровери М15/M5."
                            )
                            sent.add((uid, pair, "u"))

                    if z.lower_zone:
                        low, high = z.lower_zone
                        if low <= price <= high and (uid, pair, "l") not in sent:
                            await bot.send_message(
                                uid,
                                f"📣 {pair} е во ДОЛНАТА зона {low}-{high}.\nПровери структура."
                            )
                            sent.add((uid, pair, "l"))

            await asyncio.sleep(WATCH_INTERVAL_SECONDS)
        except Exception as e:
            print("Watcher error:", e)
            await asyncio.sleep(WATCH_INTERVAL_SECONDS)

# ============================
# WEBHOOK + AIOHTTP SERVER
# ============================

PORT = int(os.getenv("PORT", 8000))
BASE_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = BASE_URL + WEBHOOK_PATH if BASE_URL else ""

async def on_startup(app: web.Application):
    # стартува background watcher
    app["price_watcher"] = asyncio.create_task(price_watcher())
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
        print("Webhook set to:", WEBHOOK_URL)
    else:
        print("WARNING: RENDER_EXTERNAL_URL не е поставен.")

async def on_shutdown(app: web.Application):
    watcher = app.get("price_watcher")
    if watcher:
        watcher.cancel()
    await bot.delete_webhook()

def main():
    app = web.Application()
    app["bot"] = bot

    handler = SimpleRequestHandler(dp, bot)
    handler.register(app, path=WEBHOOK_PATH)

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
