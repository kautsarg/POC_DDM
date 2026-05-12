#!/usr/bin/env zsh

export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-/opt/homebrew/etc/fonts}"

mkdir -p "$PWD/.cache/matplotlib" "$PWD/.cache/fontconfig"

echo "Activated project environment:"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
echo "  Python=$(python --version 2>/dev/null)"
