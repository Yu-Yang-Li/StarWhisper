#!/usr/bin/env bash
# Install the StarWhisper skill pack into Codex and/or Cursor.
#
#   ./skills/install.sh --list
#   ./skills/install.sh
#   ./skills/install.sh --set native --target codex
set -euo pipefail

SET=all
TARGET=both
LIST=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --set)     SET="$2"; shift 2 ;;
        --target)  TARGET="$2"; shift 2 ;;
        --list)    LIST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$SET" in all|native|research) ;; *) echo "--set must be all, native or research" >&2; exit 2 ;; esac
case "$TARGET" in both|codex|cursor) ;; *) echo "--target must be both, codex or cursor" >&2; exit 2 ;; esac

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"

skills=()
for dir in "$here"/*/; do
    [ -f "${dir}SKILL.md" ] || continue
    name="$(basename "$dir")"
    case "$SET" in
        native)   case "$name" in starwhisper-*) ;; *) continue ;; esac ;;
        research) case "$name" in starwhisper-*) continue ;; esac ;;
    esac
    skills+=("$name")
done

if [ ${#skills[@]} -eq 0 ]; then
    echo "no skills matched --set $SET under $here" >&2
    exit 1
fi

if [ "$LIST" -eq 1 ]; then
    for name in "${skills[@]}"; do
        case "$name" in
            starwhisper-*) echo "native    $name" ;;
            *)             echo "research  $name" ;;
        esac
    done
    echo "${#skills[@]} skills"
    exit 0
fi

targets=()
if [ "$TARGET" = both ] || [ "$TARGET" = codex ]; then
    targets+=("$HOME/.codex/skills")
fi
if [ "$TARGET" = both ] || [ "$TARGET" = cursor ]; then
    targets+=("$HOME/.cursor/skills")
fi

for root in "${targets[@]}"; do
    if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$root"; fi
    for name in "${skills[@]}"; do
        dest="$root/$name"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "would install $name -> $dest"
            continue
        fi
        rm -rf "$dest"
        cp -R "$here/$name" "$dest"
        echo "installed $name -> $dest"
    done
done

if [ "$DRY_RUN" -eq 1 ]; then exit 0; fi

echo
echo "Installed ${#skills[@]} skills into ${#targets[@]} location(s)."
echo "Point the native skills at this checkout so they can read snclock/, explore/ and NGSS/:"
echo "  export STARWHISPER_ROOT=\"$repo\""
echo "Skills with extra Python deps: experiment-design, statistical-analysis, thesis-audit-reviewer, visual-deck-builder, papercheck."
echo "The four starwhisper-* skills are stdlib only and need nothing installed."
