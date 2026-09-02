"""Чистка распознанного текста от галлюцинаций Whisper: петли и титры.

Отдельный модуль без тяжёлых импортов — чтобы правила можно было гонять
тестами (`python3 test_textclean.py`), не поднимая rumps/MLX."""
import re

# Whisper учился на субтитрах и на хвосте тишины дописывает титры из обучающих
# данных. Ищем их ТОЛЬКО в самом конце: в середине такая фраза почти наверняка
# настоящая («продолжение следует на следующей неделе»)
TAIL_JUNK = ("продолжение следует", "субтитры сделал", "субтитры делал",
             "субтитры создавал", "субтитры и перевод", "редактор субтитров",
             "спасибо за просмотр", "подписывайтесь на канал", "ставьте лайки",
             "thanks for watching", "thank you for watching",
             "subtitles by", "amara.org")
# после титра допустимы только знаки и пробелы: буквы дальше — это речь
# («продолжение следует на следующей неделе»). Исключение — титры с подписью:
# за «субтитры сделал» идёт имя автора, там буквы ждём
TAIL_REST = re.compile(r"[\s.!?…,;:\-–—\"'»)]*")
TAIL_SIGNED = ("субтитры сделал", "субтитры делал", "субтитры создавал",
               "редактор субтитров", "subtitles by")
# Титр считаем галлюцинацией только если аудио на это намекает: сегмент
# похож на тишину, пришёл после паузы или модель в нём не уверена. Без
# данных о сегментах (старый вызов, тесты) — режем как раньше
TAIL_NO_SPEECH = 0.4   # no_speech_prob сегмента
TAIL_GAP_SEC = 1.0     # пауза перед сегментом
TAIL_LOGPROB = -1.0    # avg_logprob сегмента
# Зацикливание: кусок повторяется подряд четыре раза и больше. Кусок берём
# любой (`.`, а не `\w`), поэтому ловятся и СКЛЕЕННЫЕ повторы «ratosratos…»,
# мимо которых счёт по словам проходил насквозь
LOOP_RE = re.compile(r"(.{2,30}?)\1{3,}", re.S)
LOOP_MIN = 12  # символов в петле: короче — это «ха-ха-ха-ха», а не галлюцинация
# короткое слово человек и сам повторяет («нет нет нет нет нет», «ха-ха-ха-ха-ха»):
# такую единицу режем только от LOOP_SHORT_REPS повторов подряд
LOOP_SHORT_UNIT = 5
LOOP_SHORT_REPS = 8
LETTER = re.compile(r"[^\W\d_]")  # хоть одна буква: «1 000 000 000» — число, не петля


def _tail_is_junk(text: str, i: int, segments) -> bool:
    """Гейт по аудио для титра, начинающегося с позиции i."""
    if not segments:
        return True
    # сегмент, на который приходится начало титра: считаем по накопленной
    # длине текстов сегментов (result["text"] — их конкатенация)
    pos, prev_end, seg, gap = 0, None, None, 0.0
    for s in segments:
        t = s.get("text", "")
        end = pos + len(t)
        if i < end or s is segments[-1]:
            seg = s
            if prev_end is not None:
                gap = float(s.get("start", 0)) - prev_end
            break
        pos, prev_end = end, float(s.get("end", 0))
    if seg is None:
        return True
    return (float(seg.get("no_speech_prob", 0)) > TAIL_NO_SPEECH
            or gap > TAIL_GAP_SEC
            or float(seg.get("avg_logprob", 0)) < TAIL_LOGPROB)


def strip_loops(text: str, segments=None) -> tuple[str, list[str]]:
    """Вырезать из распознанного галлюцинации Whisper: петли и титры.

    Петля («ratos» 220 раз, «secular secular…») и хвост «Продолжение
    следует…» — не оговорка диктующего, а бред модели на тишине. Сторож ниже
    считал повторы по СЛОВАМ: склеенную петлю он не видел вовсе, а когда
    срабатывал — выбрасывал диктовку ЦЕЛИКОМ, вместе с нормальным началом
    (49 секунд речи в мусор). Поэтому режем точечно, остальное едет дальше.

    segments — result["segments"] Whisper: по no_speech_prob/паузам отличаем
    надиктованное «спасибо за просмотр» от дописанного на тишине.

    Возвращает (очищенный текст, что вырезали) — вырезанное идёт в лог."""
    cut, out, pos = [], [], 0
    for m in LOOP_RE.finditer(text):
        unit, whole = m.group(1), m.group(0)
        reps = len(whole) // len(unit)
        if len(whole) < LOOP_MIN or not LETTER.search(unit):
            continue
        if len(unit.strip()) < LOOP_SHORT_UNIT and reps < LOOP_SHORT_REPS:
            continue
        out.append(text[pos:m.start()])
        cut.append(f"«{unit.strip()}»×{reps}")
        pos = m.end()
    out.append(text[pos:])
    text = re.sub(r"\s{2,}", " ", "".join(out)).strip()
    for _ in range(len(TAIL_JUNK)):  # титров может быть несколько подряд
        low = text.lower()
        for phrase in TAIL_JUNK:
            i = low.rfind(phrase)
            # титры сидят в САМОМ хвосте: после них — знаки и точки, не больше.
            # i > 0 — на всю диктовку правило не распространяется: одну фразу
            # целиком не выбрасываем
            rest = text[i + len(phrase):]
            if (i > 0 and len(rest) <= 20
                    and (phrase in TAIL_SIGNED or TAIL_REST.fullmatch(rest))
                    and _tail_is_junk(text, i, segments)):
                cut.append(text[i:].strip())
                text = text[:i].rstrip(" \t,-–—")  # точку предложения оставляем
                break
        else:
            break
    return text.strip(), cut
