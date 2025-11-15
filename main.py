import os
import logging
from typing import Dict, Optional, Tuple

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import Message

from openai import OpenAI

# -------------------------------------------------
#  CONFIG & GLOBALS
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN env var")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

client = OpenAI(api_key=OPENAI_API_KEY)

# webhook (Render)
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RENDER_EXTERNAL_HOSTNAME")
if RENDER_EXTERNAL_URL and not RENDER_EXTERNAL_URL.startswith("http"):
    RENDER_EXTERNAL_URL = "https://" + RENDER_EXTERNAL_URL

WEBHOOK_HOST = RENDER_EXTERNAL_URL or f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', '')}"
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = (WEBHOOK_HOST + WEBHOOK_PATH) if WEBHOOK_HOST else None

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", 10000))

# Во меморија ги чуваме плановите по user_id
USER_PLANS: Dict[int, Dict] = {}


# -------------------------------------------------
#  HELPER FUNKCIJI
# -------------------------------------------------


def parse_plan(text: str) -> Optional[Dict]:
    """
    Очекуваме формат, на пример:

    /plan
    pair: EURUSD
    bias: short
    upper_zone: 1.0850-1.0870
    lower_zone: 1.0760-1.0780
    rr: 1:2
    reason: нешто...
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # првата линија е /plan
    lines = [l for l in lines if not l.lower().startswith("/plan")]

    data = {}
    for line in lines:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        data[key] = val

    required = ["pair", "bias", "upper_zone", "lower_zone"]
    if not all(k in data for k in required):
        return None

    return {
        "pair": data["pair"].upper(),
        "bias": data["bias"].lower(),
        "upper_zone": data["upper_zone"],
        "lower_zone": data["lower_zone"],
        "rr": data.get("rr", "1:2"),
        "reason": data.get("reason", ""),
    }


def parse_check(text: str) -> Optional[Dict]:
    """
    /check
    pair: EURUSD
    direction: short
    entry: 1.0860
    sl: 1.0880
    tp: 1.0820
    reason: ...
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not l.lower().startswith("/check")]

    data = {}
    for line in lines:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        data[key] = val

    required = ["pair", "direction", "entry", "sl", "tp"]
    if not all(k in data for k in required):
        return None

    try:
        entry = float(data["entry"].replace(",", "."))
        sl = float(data["sl"].replace(",", "."))
        tp = float(data["tp"].replace(",", "."))
    except ValueError:
        return None

    return {
        "pair": data["pair"].upper(),
        "direction": data["direction"].lower(),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "reason": data.get("reason", ""),
    }


def calc_rr(entry: float, sl: float, tp: float) -> Optional[Tuple[float, float, float]]:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return risk, reward, reward / risk


async def ai_analyze_plan(plan: Dict) -> str:
    upper = plan["upper_zone"]
    lower = plan["lower_zone"]

    prompt = f"""
Ти си FX/crypto price action ментор. Корисникот е почетник.

Анализирај го планот:
- пар: {plan['pair']}
- bias: {plan['bias']}
- горна зона: {upper}
- долна зона: {lower}
- RR преференција: {plan['rr']}
- причина: {plan['reason']}

Објасни:
1) Дали bias има логика (теоретски, без да знаеш точни цени).
2) Како би изгледале овие зони (supply/demand, support/resistance) во нормален чарт.
3) Како отприлика би размислувал за entry, SL и TP (без да даваш точни цени).
4) Што е добро во планот и што е потенцијален ризик.
5) Објаснувај поедноставно, на македонски, како ментор на почетник.
6) Не давај директен совет: само објаснувај логика.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return resp.choices[0].message.content


async def ai_check_setup(check: Dict, zone: Dict) -> str:
    rr = calc_rr(check["entry"], check["sl"], check["tp"])
    if rr:
        risk, reward, rr_val = rr
        rr_line = f"Ризик: {risk:.5f}, награда: {reward:.5f}, RR≈{rr_val:.2f}"
    else:
        rr_line = "Не може да се пресмета RR."

    prompt = f"""
FX/crypto ментор, провери го следниов сетап.

PAIR: {check['pair']}
Direction: {check['direction']}
Entry: {check['entry']}
SL: {check['sl']}
TP: {check['tp']}
{rr_line}

План/зони од корисникот:
- Bias: {zone.get('bias')}
- Upper zone: {zone.get('upper_zone')}
- Lower zone: {zone.get('lower_zone')}
- Reason: {zone.get('reason')}

Објасни:
1) Дали насоката (long/short) има смисла со bias.
2) Дали позицијата на SL и TP изгледа логично во однос на зоните (теоретски).
3) Дали RR е здрав за еден почетник.
4) На што би внимава(л) ти, што може да појде наопаку.
5) Сè на македонски, едукативно, без директни сигнали.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return resp.choices[0].message.content


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

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )

    return resp.choices[0].message.content


# -------------------------------------------------
#  TELEGRAM HANDLERS
# -------------------------------------------------


@dp.message_handler(commands=["start"])
async def cmd_start(m: Message):
    await m.answer(
        "👋 Здраво, јас сум FX Mentor Bot.\n\n"
        "Можеш да:\n"
        "• Направиш план со /plan\n"
        "• Ми пратиш конкретен сетап со /check\n"
        "• Да ги видиш активните зони со /zones\n"
        "• Да ги избришеш зоните со /clear\n"
        "• Да ми пратиш screenshot од чарт со /chart (во caption) за едукативна анализа.\n\n"
        "Сè е само за учење, НЕ е финансиски совет. 😊"
    )


@dp.message_handler(commands=["help"])
async def cmd_help(m: Message):
    await m.answer(
        "Еве примери како да ме користиш:\n\n"
        "📌 /plan пример:\n"
        "/plan\n"
        "pair: EURUSD\n"
        "bias: short\n"
        "upper_zone: 1.0850-1.0870\n"
        "lower_zone: 1.0760-1.0780\n"
        "rr: 1:2\n"
        "reason: H4 downtrend, структура надолу\n\n"
        "📌 /check пример:\n"
        "/check\n"
        "pair: EURUSD\n"
        "direction: short\n"
        "entry: 1.0860\n"
        "sl: 1.0880\n"
        "tp: 1.0820\n"
        "reason: ретест на зона, M15 rejection\n\n"
        "📌 /chart пример (како caption на слика):\n"
        "/chart\n"
        "pair: BTCUSD\n"
        "tf: H1\n"
        "bias: long\n"
        "plan: гледам uptrend, можен retest на зона"
    )


@dp.message_handler(commands=["plan"])
async def cmd_plan(m: Message):
    plan = parse_plan(m.text)
    if not plan:
        await m.answer(
            "Форматот на /plan не е добар.\n"
            "Пример:\n\n"
            "/plan\n"
            "pair: EURUSD\n"
            "bias: short\n"
            "upper_zone: 1.0850-1.0870\n"
            "lower_zone: 1.0760-1.0780\n"
            "rr: 1:2\n"
            "reason: H4 downtrend..."
        )
        return

    USER_PLANS[m.from_user.id] = plan

    await m.answer("✅ Планот е зачуван. Сега ќе направам едукативна анализа...")
    analysis = await ai_analyze_plan(plan)
    await m.answer(analysis)


@dp.message_handler(commands=["zones"])
async def cmd_zones(m: Message):
    plan = USER_PLANS.get(m.from_user.id)
    if not plan:
        await m.answer("Немаш активен план. Користи /plan за да внесеш зони.")
        return

    txt = (
        f"📌 Активен план:\n\n"
        f"Pair: {plan['pair']}\n"
        f"Bias: {plan['bias']}\n"
        f"Upper zone: {plan['upper_zone']}\n"
        f"Lower zone: {plan['lower_zone']}\n"
        f"RR: {plan['rr']}\n"
        f"Reason: {plan['reason']}"
    )
    await m.answer(txt)


@dp.message_handler(commands=["clear"])
async def cmd_clear(m: Message):
    if m.from_user.id in USER_PLANS:
        USER_PLANS.pop(m.from_user.id)
        await m.answer("🧹 Ги избришав сите зони/планови за тебе.")
    else:
        await m.answer("Немаш активни зони кои треба да се бришат.")


@dp.message_handler(commands=["check"])
async def cmd_check(m: Message):
    plan = USER_PLANS.get(m.from_user.id)
    if not plan:
        await m.answer("Немаш активен план. Прво користи /plan, па после /check.")
        return

    check = parse_check(m.text)
    if not check:
        await m.answer(
            "Форматот на /check не е добар.\n"
            "Пример:\n\n"
            "/check\n"
            "pair: EURUSD\n"
            "direction: short\n"
            "entry: 1.0860\n"
            "sl: 1.0880\n"
            "tp: 1.0820\n"
            "reason: ретест на зона, M15 rejection"
        )
        return

    await m.answer("✅ Го примив сетапот, правам анализа...")
    analysis = await ai_check_setup(check, plan)
    await m.answer(analysis)


# ---------------  /chart  -------------------------


@dp.message_handler(lambda m: (m.caption and m.caption.lower().startswith("/chart")) or (m.text and m.text.lower().startswith("/chart")),
                    content_types=types.ContentTypes.ANY)
async def cmd_chart(m: Message):
    """
    /chart команда – се користи како caption на слика.
    """
    raw_text = (m.caption or m.text or "").strip()

    # ако нема слика, враќаме инструкции
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

    caption = m.caption or m.text or ""
    await m.answer("⏳ Го читам чартот, секундна...")

    try:
        analysis = await ai_analyze_chart_image(file_url, caption)
        await m.answer(analysis)
    except Exception as e:
        logging.exception("Chart analysis error: %s", e)
        await m.answer("❌ Настана грешка при анализа на чартот. Провери дали сликата е јасна и пробај повторно.")


# -------------------------------------------------
#  WEBHOOK START / STOP
# -------------------------------------------------


async def on_startup(dp: Dispatcher):
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
        logging.info(f"Webhook set to: {WEBHOOK_URL}")
    else:
        logging.warning("WEBHOOK_URL не е сетнат (нема RENDER_EXTERNAL_URL)")


async def on_shutdown(dp: Dispatcher):
    logging.warning("Shutting down..")
    await bot.delete_webhook()
    await bot.session.close()
    logging.warning("Bye!")


if __name__ == "__main__":
    logging.info("Starting webhook bot...")
    executor.start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
