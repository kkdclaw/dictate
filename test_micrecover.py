"""Тесты починки уснувшего микрофона: `python3 test_micrecover.py`.

Sony WH-1000XM5 в спящем HFP отдают не нули, а свой шумовой пол (RMS 0.0005,
peak 0.0024 — замер 10.09.2026), поэтому детекторы тишины считают линк живым
и не чинят его. Чиним по результату записи — здесь проверяем, что чиним ровно
там, где надо, и не дёргаем Bluetooth почём зря."""
import time
import dictate as d

calls = []


def setup(*, opened_ago=100.0, link_reopen_ago=1e6, recording=False):
    """Поток открыт opened_ago секунд назад, запись только что кончилась.
    Шумовой пол спящего линка продолжает идти — last_signal всегда свежий."""
    calls.clear()
    d.reopen_stream = lambda **kw: calls.append(kw)
    now = time.time()
    d.recording = recording
    d.stream_holder.clear()
    d.stream_holder["opened"] = now - opened_ago
    d.stream_holder["last_signal"] = now + 0.05
    d.stream_holder["link_reopen"] = now - link_reopen_ago
    return now  # rec_end


REOPEN = [{"follow_default": True, "force": True}]

# спящий линк, человек говорил 3 с — переоткрываем и запоминаем время
rec_end = setup()
d.recover_from_empty_capture(3.0, rec_end)
assert calls == REOPEN, calls
assert d.stream_holder["link_reopen"] >= rec_end, "гейт не взведён"

# то же самое минутой раньше уже было и не помогло — второй щелчок не делаем
rec_end = setup(link_reopen_ago=5.0)
d.recover_from_empty_capture(3.0, rec_end)
assert calls == [], calls

# гейт истёк — пробуем снова
rec_end = setup(link_reopen_ago=d.LINK_REOPEN_GATE_SEC + 1)
d.recover_from_empty_capture(3.0, rec_end)
assert calls == REOPEN, calls

# случайный тычок хоткея: писать нечего, линк не трогаем
rec_end = setup()
d.recover_from_empty_capture(0.4, rec_end)
assert calls == [], calls

# задание ждало очереди за Whisper, а поток за это время уже переоткрыли
rec_end = setup()
d.stream_holder["opened"] = rec_end + 1
d.recover_from_empty_capture(3.0, rec_end)
assert calls == [], calls

# идёт новая запись — рвать поток нельзя: дыра в chunks и потерянные слова
rec_end = setup(recording=True)
d.recover_from_empty_capture(3.0, rec_end)
assert calls == [], calls

# шумовой пол не выдаём за «микрофон ожил»: для пустого захвата сигналу не верим,
# для мёртвых нулей — верим, там last_signal двигают только звук и открытие потока
rec_end = setup()
assert d.stream_refreshed_since(rec_end, False) is False
assert d.stream_refreshed_since(rec_end, True) is True

# микрофон исчез совсем, пока мы собирались чинить — не падаем
def boom(**kw):
    raise d.NoMicrophone("нет входных устройств")


rec_end = setup()
d.reopen_stream = boom
d.recover_from_empty_capture(3.0, rec_end)
assert d.STATE["mic"] == "нет — подключи микрофон"

# порог пустого захвата держим между реальными замерами, а не на глаз
assert 0.0024 < d.EMPTY_CAPTURE_PEAK, "спящие XM5 обязаны считаться пустым захватом"
assert 0.076 > d.EMPTY_CAPTURE_PEAK, "живая тишина (лог 10.09) — не пустой захват"

print("ok")
