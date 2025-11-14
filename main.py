# ============================
# FX Mentor Bot (main.py)
# ============================

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
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
        try: return float(v)
        except: return None

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
    if not entry or not sl or not tp: return None
    risk = abs(entry - sl)
    reward = abs(entry - tp)
    if risk == 0: return None
    return risk, reward, reward/risk


async def fetch_price(symbol: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=10) as s:
            r = await s.get("https://api.twelvedata.com/price",
                params={"symbol": symbol, "apikey": TWELVEDATA_API_KEY}
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
        input=[{"role": "user",
                "content":[{"type":"input_text","text": prompt}]}]
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
        input=[{"role": "user",
                "content":[{"type":"input_text","text": prompt}]}]
    )

    return r.output[0].content[0].text


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
        parse_mode="Markdown"
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


# ============================
# PRICE WATCHER
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

                    # upper zone
                    if z.upper_zone:
                        low, high = z.upper_zone
                        if low <= price <= high and (uid, pair, "u") not in sent:
                            await bot.send_message(
                                uid,
                                f"📣 {pair} е во ГОРНАТА зона {low}-{high}.\nПровери М15/M5."
                            )
                            sent.add((uid, pair, "u"))

                    # lower zone
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
# MAIN
# ============================

async def main():
    asyncio.create_task(price_watcher())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
