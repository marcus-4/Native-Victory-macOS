//! `victory run --list-imports` on the native host, against the probe
//! fixture (no game data): it loads and binds, lists, and never runs.
#![cfg(all(target_os = "macos", target_arch = "x86_64"))]

use std::path::Path;
use std::process::Command;

#[test]
fn list_imports_binds_the_probe_and_does_not_run_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let exe = Path::new(env!("CARGO_TARGET_TMPDIR")).join("list-imports-probe.exe");
    let built = Command::new(root.join("tools/macos/build-native-probe.sh"))
        .arg(&exe)
        .status()
        .unwrap();
    // Exit 3: the probe's cross tools are not installed (see native_probe.rs).
    if built.code() == Some(3) && std::env::var_os("VICTORY_REQUIRE_PROBE").is_none() {
        eprintln!("skipped: the native probe's cross tools are not installed (brew install llvm lld mingw-w64)");
        return;
    }
    assert!(built.success());

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .args(["run", "--list-imports"])
        .arg(&exe)
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{out}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        out.contains("native (hle)   xgameruntime.dll!XGameRuntimeInitialize"),
        "{out}"
    );
    assert!(
        out.contains("native (host)  KERNEL32.dll!VirtualAlloc"),
        "{out}"
    );
    assert!(
        out.contains("STUB           victory-probe-host.dll!ProbeNotImplemented"),
        "{out}"
    );
    // The probe's host DLL is never built, so its folder has no such file;
    // the DLL the probe does ship beside it is found.
    assert!(
        out.contains("missing dll    victory-probe-host.dll: not in"),
        "{out}"
    );
    assert!(
        !out.contains("missing dll    victory-probe-dll.dll"),
        "{out}"
    );
    assert!(out.contains("not executing"), "{out}");
    // The probe's entry point would have printed a "title exited" line.
    assert!(!out.contains("title exited"), "{out}");
}
