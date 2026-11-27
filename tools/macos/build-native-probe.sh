#!/bin/sh
# Builds the native host's stand-in title (tests/fixtures/native-probe) with
# the same clang and lld as the gnullvm cross build (gnullvm-cc.sh).
# Usage: tools/macos/build-native-probe.sh [output.exe]
# The default output is target/fixtures/native-probe.exe; victory-probe-dll.dll
# is written beside it. Prints the path.
set -e
cd "$(dirname "$0")/../.."
SRC=tests/fixtures/native-probe
OUT="${1:-target/fixtures/native-probe.exe}"
LLVM="${LLVM_PREFIX:-/opt/homebrew/opt/llvm}"
LLD="${LLD_PREFIX:-/opt/homebrew/opt/lld}"
MINGW="${MINGW_PREFIX:-/opt/homebrew/opt/mingw-w64/toolchain-x86_64}"
# Name the cross tool that is missing rather than fail on it half way. Exit 3
# says only that (the tests that build the probe skip on it); 1 is any other failure.
for tool in "$LLVM/bin/llvm-dlltool" "$LLVM/bin/clang" "$LLD/bin/ld.lld" "$MINGW/x86_64-w64-mingw32"; do
    if [ ! -e "$tool" ]; then
        echo "build-native-probe.sh: $tool is missing. Install the cross tools with" \
            "'brew install llvm lld mingw-w64', or set LLVM_PREFIX, LLD_PREFIX or MINGW_PREFIX." >&2
        exit 3
    fi
done
WORK="$(dirname "$OUT")/native-probe-libs"
mkdir -p "$WORK"
for def in xgameruntime victory-probe-host victory-probe-dll ucrtbase-probe ucrt-math; do
    "$LLVM/bin/llvm-dlltool" -m i386:x86-64 -d "$SRC/$def.def" -l "$WORK/lib$def.a"
done
# Import libraries for the network, media and audio DLLs (probe_net.c).
for def in probe-ws2 probe-winhttp probe-mfplat probe-xaudio2 probe-d3d12; do
    "$LLVM/bin/llvm-dlltool" -m i386:x86-64 -d "$SRC/$def.def" -l "$WORK/lib$def.a"
done
# The DLL the probe imports, built first: it goes beside the probe, as a
# title's DLLs do. Its own preferred base is not the probe's.
tools/macos/gnullvm-cc.sh -shared -O1 -ffreestanding -fno-builtin -fno-stack-protector \
    -fms-extensions -fasynchronous-unwind-tables -fexceptions -Xclang -fasync-exceptions -fno-emulated-tls -nostdlib \
    -Wl,--entry,DllMain -Wl,--subsystem,console -Wl,--dynamicbase \
    -Wl,--image-base,0x150000000 \
    -L"$WORK" "$SRC/probe_dll.c" "$SRC/victory-probe-dll.def" -lkernel32 -o "$(dirname "$OUT")/victory-probe-dll.dll"
# No CRT: the probe's entry point is probe_main, which the host calls itself.
# --dynamicbase gives it base relocations; the preferred base is above 4 GiB
# like a console title's.
tools/macos/gnullvm-cc.sh -O1 -ffreestanding -fno-builtin -fno-stack-protector \
    -fms-extensions -fasynchronous-unwind-tables -fexceptions -Xclang -fasync-exceptions -fno-emulated-tls -nostdlib \
    -Wl,--entry,probe_main -Wl,--subsystem,console -Wl,--dynamicbase \
    -Wl,--image-base,0x140000000 -Wl,--export-all-symbols \
    -L"$WORK" "$SRC/probe.c" "$SRC/probe_net.c" -lxgameruntime -lvictory-probe-host -lvictory-probe-dll -lucrtbase-probe -lucrt-math \
    -lprobe-ws2 -lprobe-winhttp -lprobe-mfplat -lprobe-xaudio2 -lprobe-d3d12 -lkernel32 -o "$OUT"
echo "$OUT"
