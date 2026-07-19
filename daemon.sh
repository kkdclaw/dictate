#!/bin/zsh
# Установка/удаление launchd-агента диктовки.
#   ./daemon.sh install   — создать агент, запустить, включить автозапуск при входе
#   ./daemon.sh uninstall — остановить и убрать
#   ./daemon.sh restart   — перезапустить (подхватить новую версию скрипта)
set -e
LABEL="com.kkd.dictate"
DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

case "$1" in
install)
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$DIR/.venv/bin/python3</string>
    <string>$DIR/dictate.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DIR/dictate.log</string>
  <key>StandardErrorPath</key><string>$DIR/dictate.log</string>
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
  launchctl kickstart -k "gui/$UID/$LABEL"
  echo "Агент перезапущен."
  ;;
*)
  echo "Использование: $0 install|uninstall|restart"; exit 1 ;;
esac
