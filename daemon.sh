#!/bin/zsh
# Установка/удаление launchd-агента диктовки.
#   ./daemon.sh install   — создать агент, запустить, включить автозапуск при входе
#   ./daemon.sh uninstall — остановить и убрать
#   ./daemon.sh restart   — перезапустить (подхватить новую версию скрипта)
set -e
LABEL="com.kkd.dictate"
DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
# путь попадает в XML plist: & < > в имени папки иначе делают его невалидным
XDIR="${DIR//&/&amp;}"; XDIR="${XDIR//</&lt;}"; XDIR="${XDIR//>/&gt;}"

case "$1" in
install)
  if [ ! -x "$DIR/.venv/bin/python3" ]; then
    echo "Нет $DIR/.venv/bin/python3 — сначала выполни: uv sync" >&2
    exit 1
  fi
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$XDIR/.venv/bin/python3</string>
    <string>$XDIR/dictate.py</string>
  </array>
  <key>WorkingDirectory</key><string>$XDIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$XDIR/dictate.log</string>
  <key>StandardErrorPath</key><string>$XDIR/dictate.log</string>
</dict></plist>
EOF
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$UID" "$PLIST"
  echo "Агент установлен и запущен. Лог: $DIR/dictate.log"
  ;;
uninstall)
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Агент остановлен и удалён."
  ;;
restart)
  # лог растёт бесконечно (tqdm-строки при каждом старте) — подрезаем хвостом
  if [ -f "$DIR/dictate.log" ] && [ "$(stat -f%z "$DIR/dictate.log")" -gt 5000000 ]; then
    tail -c 1000000 "$DIR/dictate.log" > "$DIR/dictate.log.tmp" \
      && mv "$DIR/dictate.log.tmp" "$DIR/dictate.log"
    echo "Лог подрезан до 1 МБ."
  fi
  if ! launchctl kickstart -k "gui/$UID/$LABEL" 2>/dev/null; then
    echo "Агент не загружен — загружаю заново."
    launchctl bootstrap "gui/$UID" "$PLIST"
  fi
  echo "Агент перезапущен."
  ;;
*)
  echo "Использование: $0 install|uninstall|restart"; exit 1 ;;
esac
