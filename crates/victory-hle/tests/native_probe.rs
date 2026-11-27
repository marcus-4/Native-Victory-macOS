//! The native macOS host end to end, with no game data: a freestanding PE
//! (tests/fixtures/native-probe, built by tools/macos/build-native-probe.sh)
//! is mapped into this process, relocated, bound against victory-hle's
//! registry and run, under Rosetta when the test binary is x86-64.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use victory_loader::native::NativeImage;

/// The probe, built once; `None` when the cross tools it needs are not installed
/// (`build-native-probe.sh` exits 3 and names the one missing).
fn built_probe() -> Option<&'static Path> {
    static BUILT: OnceLock<Option<PathBuf>> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("native-probe.exe");
            let status = Command::new(root.join("tools/macos/build-native-probe.sh"))
                .arg(&out)
                .status()
                .expect("running build-native-probe.sh");
            if status.code() == Some(3) && std::env::var_os("VICTORY_REQUIRE_PROBE").is_none() {
                return None;
            }
            assert!(status.success(), "building the native probe failed");
            Some(out)
        })
        .as_deref()
}

/// Whether the probe could be built. Without the cross tools the test says it
/// is skipped and passes (`VICTORY_REQUIRE_PROBE=1` makes that a failure), as
/// the GPU tests do without the Metal Shader Converter.
fn have_probe() -> bool {
    let have = built_probe().is_some();
    if !have {
        eprintln!("skipped: the native probe's cross tools are not installed (brew install llvm lld mingw-w64)");
    }
    have
}

fn probe_exe() -> &'static Path {
    built_probe().expect("the native probe")
}

/// Maps the probe at an address nothing is mapped at (found, not assumed:
/// where a process keeps its own mappings differs between x86-64 and arm64
/// hosts), and returns that address too.
fn map_elsewhere(bytes: &[u8]) -> (NativeImage, usize) {
    use victory_platform::mem;
    // The probe tests run in parallel and would pick the same free address.
    static LOOKING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = LOOKING.lock().unwrap_or_else(|e| e.into_inner());
    // Other tests' threads allocate meanwhile and can take the address
    // between the release and the map: look again.
    for _ in 0..50 {
        // SAFETY: reserved and released at once; the image maps there next.
        let at = unsafe {
            let p = mem::reserve(None, 1 << 20).unwrap();
            mem::release(p, 1 << 20).unwrap();
            p as usize
        };
        if let Ok(image) = NativeImage::map(bytes, Some(at)) {
            return (image, at);
        }
    }
    panic!("no free address to map the probe at");
}

fn probe_bytes() -> Vec<u8> {
    std::fs::read(probe_exe()).expect("reading the probe")
}

#[test]
fn an_image_maps_and_relocates_at_another_base() {
    if !have_probe() {
        return;
    }
    let bytes = probe_bytes();
    let preferred = NativeImage::map(&bytes, None).unwrap();
    let (moved, at) = map_elsewhere(&bytes);
    assert_eq!(moved.base(), at);
    // The TLS directory is all absolute addresses, so each is rebased.
    let (a, b) = (preferred.tls().unwrap(), moved.tls().unwrap());
    assert_eq!(a.start - preferred.base(), b.start - moved.base());
    assert_eq!(a.index_at - preferred.base(), b.index_at - moved.base());
    assert!(b.end > b.start && b.end < moved.base() + moved.len());
    // Sections landed where their RVAs say: the entry point's bytes are code.
    let entry = moved.entry().unwrap();
    assert_eq!(moved.export("probe_main"), Some(entry));
    // SAFETY: inside the mapping.
    let first = unsafe { (entry as *const u8).read() };
    assert_ne!(first, 0);
    // SAFETY: nothing runs from these images.
    unsafe {
        preferred.unmap();
        moved.unmap();
    }
}

#[test]
fn finalize_gives_sections_their_protections() {
    if !have_probe() {
        return;
    }
    let (image, _) = map_elsewhere(&probe_bytes());
    image.finalize().unwrap();
    use victory_platform::mem::{Protect, query};
    let entry = image.entry().unwrap();
    assert_eq!(query(entry).unwrap().protect, Protect::READ_EXECUTE);
    // The headers share a page with .text where the host's page is larger
    // than the PE's 4 KiB, and then take its protection.
    let headers = query(image.base()).unwrap().protect;
    if victory_platform::mem::page_size() == 4096 {
        assert_eq!(headers, Protect::READ, "headers");
    } else {
        assert_eq!(headers, Protect::READ_EXECUTE);
    }
    let tls = image.tls().unwrap();
    assert_eq!(query(tls.start).unwrap().protect, Protect::READ_WRITE);
    // SAFETY: nothing runs from this image.
    unsafe { image.unmap() };
}

#[cfg(target_arch = "x86_64")]
mod run {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use victory_hle::HleResolver;
    use victory_loader::native::dll::{self, TitleDll, WithTitleDlls};
    use victory_loader::native::{self, call_win64, declare_win64, teb};
    use victory_loader::{ImportResolver, Provenance, Resolution, UnimplementedPolicy};
    use victory_pe::ImportSymbol;

    /// Every check in probe.c passed.
    const PROBE_ALL: u32 = 0xFFFF;

    // The test's own "DLL": `victory-probe-host.dll`.

    extern "C" fn probe_sum8(
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        e: u64,
        f: u64,
        g: u64,
        h: u64,
    ) -> u64 {
        // A System V function that sees all eight, however the Windows
        // caller passed them (four in registers, four on its stack).
        a + b + c + d + e + f + g + h
    }

    // Windows ABI: floats arrive in xmm0, the integer in edx.
    extern "win64" fn probe_scale(x: f64, k: i32) -> f64 {
        // Host code with a std call: GS must be the host's here.
        let _ = std::thread::current().id();
        x * f64::from(k)
    }

    static CALLBACKS: AtomicU32 = AtomicU32::new(0);
    /// A process has one main image (the PEB names it), so runs take turns.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    extern "C" fn probe_call(callback: usize, a: u64, b: u64) -> u64 {
        CALLBACKS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the probe passes one of its own functions of two arguments.
        unsafe { call_win64(callback, &[a, b]) }
    }

    struct ProbeResolver(HleResolver);

    impl ImportResolver for ProbeResolver {
        fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
            if !dll.eq_ignore_ascii_case("victory-probe-host.dll") {
                return self.0.resolve(dll, symbol);
            }
            let address = match symbol.to_string().as_str() {
                "ProbeSum8" => probe_sum8 as *const () as usize,
                "ProbeScale" => {
                    declare_win64(probe_scale as *const () as usize);
                    probe_scale as *const () as usize
                }
                "ProbeCall" => probe_call as *const () as usize,
                _ => return Resolution::Missing,
            };
            Resolution::Found {
                address,
                provenance: Provenance::Host,
            }
        }
    }

    /// `victory-probe-dll.dll`, which the probe imports and the build puts
    /// beside it: mapped, bound and started (`DllMain`, TLS callback) once,
    /// as the loader does before the executable's first thread.
    fn title_dlls() -> &'static [&'static TitleDll] {
        static DLLS: std::sync::OnceLock<Vec<&'static TitleDll>> = std::sync::OnceLock::new();
        DLLS.get_or_init(|| {
            let info = victory_pe::ImageInfo::parse(&probe_bytes()).unwrap();
            let dir = probe_exe().parent().unwrap();
            let dlls = dll::map_title_dlls(&info, dir, &|_| true).unwrap();
            assert_eq!(dlls.len(), 1, "the probe imports one title DLL");
            let hle = HleResolver::default();
            let resolver = WithTitleDlls { inner: &hle };
            for d in &dlls {
                let report =
                    native::bind_imports(&d.image, &resolver, UnimplementedPolicy::Trap).unwrap();
                assert!(report.missing.is_empty(), "{:?}", report.missing);
                d.image.finalize().unwrap();
                victory_hle::runtime::set_image_path(d.image.base(), &d.path);
            }
            let started = dlls.clone();
            std::thread::spawn(move || {
                for d in started {
                    // SAFETY: the probe's own DLL, bound and finalized above.
                    assert!(unsafe { native::attach_dll(&d.image, 1, 1) });
                    native::register_dll_thread_notices(&d.image);
                }
            })
            .join()
            .unwrap();
            dlls
        })
    }

    /// Maps, registers TLS, binds and finalizes the probe.
    fn load(relocated: bool, policy: UnimplementedPolicy) -> (&'static NativeImage, usize) {
        title_dlls();
        let image = if relocated {
            map_elsewhere(&probe_bytes()).0
        } else {
            NativeImage::map(&probe_bytes(), None).unwrap()
        };
        if let Some(tls) = image.tls() {
            teb::register_tls(tls);
        }
        let probe = ProbeResolver(HleResolver::default());
        let report = native::bind_imports(&image, &WithTitleDlls { inner: &probe }, policy).unwrap();
        // Everything but the one import nobody implements is bound.
        assert_eq!(report.missing.len(), 1, "{:?}", report.missing);
        assert_eq!(report.missing[0].symbol, "ProbeNotImplemented");
        assert!(report.hle >= 1 && report.host >= 10, "{report:?}");
        // Its imports of the title's DLL come from the DLL itself.
        assert_eq!(report.title, 9, "{report:?}");
        image.finalize().unwrap();
        native::register_thread_callbacks(&image);
        let missing = image.export("probe_missing").unwrap();
        (Box::leak(Box::new(image)), missing)
    }

    /// Runs one of the probe's check functions on a fresh thread (so its
    /// own TEB), and returns the bits it set.
    fn run_probe(export: &str, arg: u64) -> u32 {
        let (image, _) = load(true, UnimplementedPolicy::Trap);
        let f = image.export(export).unwrap();
        std::thread::spawn(move || {
            // SAFETY: a function of the probe taking one argument.
            unsafe { call_win64(f, &[arg]) as u32 }
        })
        .join()
        .unwrap()
    }

    #[track_caller]
    fn assert_all(result: u32, all: u32, what: &str) {
        assert_eq!(result, all, "{what}: missing bits {:#06x}", all & !result);
    }

    #[test]
    fn threads_notices_suspension_and_context() {
        if !have_probe() {
            return;
        }
        assert_all(run_probe("probe_threads", 0), 0xFF, "probe_threads");
    }

    #[test]
    fn waits_critical_sections_srw_locks_and_condition_variables() {
        if !have_probe() {
            return;
        }
        // Bits 8 and up of probe_threads' enum belong to probe_sync.
        assert_all(run_probe("probe_sync", 0) >> 8, 0x7FFFF >> 8, "probe_sync");
    }

    #[test]
    fn files_open_read_seek_find_and_write() {
        if !have_probe() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "hello, victory\n").unwrap();
        let wide: Vec<u16> = dir
            .path()
            .to_str()
            .unwrap()
            .encode_utf16()
            .chain([0])
            .collect();
        let result = run_probe("probe_files", wide.as_ptr() as u64);
        assert_all(result, 0x3FF, "probe_files");
        assert_eq!(
            std::fs::read(dir.path().join("written.bin")).unwrap(),
            b"abcd"
        );
    }

    /// Winsock by ordinal (a loopback echo, off-machine refused), WinHTTP set
    /// up and never sent, Media Foundation making nothing, and XAudio2
    /// ordinal 1 played through COM vtables of host stubs, under Rosetta.
    #[test]
    fn network_media_and_audio_imports_work() {
        if !have_probe() {
            return;
        }
        assert_all(run_probe("probe_netaudio", 0), 0xFFF, "probe_netaudio");
    }

    /// The console device from `D3D12XboxCreateDevice`, called from title code through its bridged
    /// table (Windows x64 registers and stack, `this` still the object's address).
    #[test]
    fn the_d3d12_device_and_its_fence_are_called_through_bridged_tables() {
        if !have_probe() {
            return;
        }
        victory_gpu::object::set_bridge(victory_gpu::object::Bridge {
            is_bridged: native::is_bridged,
            copy_of: native::bridged_copy_of,
            // SAFETY: forwarded: a live object of ours with such a table.
            expose: |this, len, area| unsafe { native::expose_object(this, len, area) },
            declare_win64: native::declare_win64,
        });
        assert_all(run_probe("probe_d3d12", 0), 0xFF, "probe_d3d12");
    }

    #[test]
    fn the_c_runtime_serves_a_title_that_links_it() {
        if !have_probe() {
            return;
        }
        // The probe imports `ucrtbase.dll` and `api-ms-win-crt-math-l1-1-0.dll`:
        // strings, math (doubles in xmm registers), printf and scanf over a
        // `va_list`, qsort with a title comparator, files, heap, time,
        // initializer and exit tables, signals, threads and errno.
        let dir = tempfile::tempdir().unwrap();
        let path: Vec<u8> = dir.path().to_str().unwrap().bytes().chain([0]).collect();
        let result = run_probe("probe_ucrt", path.as_ptr() as u64);
        assert_all(result, 0x1FFFF, "probe_ucrt");
        assert!(
            !dir.path().join("ucrt.bin").exists(),
            "the probe removes its file"
        );
    }

    #[test]
    fn exceptions_reach_seh_scopes_and_vectored_handlers() {
        if !have_probe() {
            return;
        }
        // The probe's vectored handler hears every thread's exceptions.
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        assert_all(run_probe("probe_seh", 0), 0x1_FFFF, "probe_seh");
    }

    /// The DLL the title ships: its `DllMain` and TLS callback ran in order
    /// before the title's code, its static TLS is per thread, threads are
    /// announced to it, an exception raised in it unwinds through its own
    /// `__finally` to a handler in the executable (and the other way round),
    /// and the title finds it by name.
    #[test]
    fn a_title_dll_has_its_own_main_tls_unwinding_and_name() {
        if !have_probe() {
            return;
        }
        // It raises exceptions, which another probe's vectored handler would hear.
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        assert_all(run_probe("probe_dll", 0), 0xFFF, "probe_dll");
    }

    #[test]
    fn get_proc_address_follows_title_forwarders() {
        assert_all(run_probe("probe_forwarders", 0), 0x3F, "probe_forwarders");
    }

    /// A title's start-up asks for `kernel32.dll` and an API set by name and
    /// looks functions up in them: both answer with a stable handle, a
    /// callable entry (called here through the pointer) and NULL with
    /// `ERROR_PROC_NOT_FOUND` for an unknown name.
    #[test]
    fn system_dll_handles_resolve_functions_through_the_binder() {
        if !have_probe() {
            return;
        }
        assert_all(run_probe("probe_modules", 0), 0x7FF, "probe_modules");
    }

    /// An HLE object the title got from `QueryApiImpl` is called through its
    /// method table from title code (every slot a bridge stub, `this` still
    /// the object), a slot past the table's end answers `E_NOTIMPL`, and our
    /// runtime calls a title callback back with GS = the TEB.
    #[test]
    fn hle_objects_are_callable_through_their_method_tables() {
        if !have_probe() {
            return;
        }
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        assert_all(run_probe("probe_vtable", 0), 0x1FF, "probe_vtable");
    }

    #[test]
    fn the_probe_runs_under_rosetta_and_passes_every_check() {
        if !have_probe() {
            return;
        }
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (image, _) = load(false, UnimplementedPolicy::Trap);
        // SAFETY: the probe is our own code; imports are bound.
        let result = unsafe { native::run_entry(image) }.unwrap();
        assert_eq!(
            result,
            PROBE_ALL,
            "failed checks (bit set = passed): {result:#06x}, missing {:#06x}",
            PROBE_ALL & !result
        );
        assert!(CALLBACKS.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn the_probe_also_runs_relocated() {
        if !have_probe() {
            return;
        }
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let (image, _) = load(true, UnimplementedPolicy::Trap);
        assert_ne!(image.base(), image.info().image_base as usize);
        // SAFETY: as above.
        let result = unsafe { native::run_entry(image) }.unwrap();
        assert_eq!(result, PROBE_ALL, "missing {:#06x}", PROBE_ALL & !result);
    }

    #[test]
    fn an_unimplemented_import_returns_zero_when_asked_to() {
        if !have_probe() {
            return;
        }
        let (_, missing) = load(false, UnimplementedPolicy::ReturnZero);
        // SAFETY: a function of one argument in the bound probe.
        assert_eq!(unsafe { call_win64(missing, &[0]) }, 0);
    }

    /// The bridge keeps what the Windows ABI makes the callee keep: here a
    /// host function that clobbers `rdi`, `rsi` and `xmm6` (free to it under
    /// System V) is called from Windows-ABI code that expects them back.
    #[test]
    fn the_bridge_preserves_windows_callee_saved_registers() {
        if !have_probe() {
            return;
        }
        extern "C" fn clobber() -> u64 {
            // SAFETY: clobbers registers the System V ABI lets us.
            unsafe {
                std::arch::asm!(
                    "mov rdi, 0x1111",
                    "mov rsi, 0x2222",
                    "pxor xmm6, xmm6",
                    out("rdi") _, out("rsi") _, out("xmm6") _,
                );
            }
            5
        }
        // Title-style code: set rdi, rsi, xmm6, call through a bridge stub,
        // and return their sum if they survived.
        //   push rdi; push rsi; sub rsp,40; movq xmm6, rax; ...
        // Written as machine code because it must follow the Windows ABI:
        let stub = {
            let mut block = native::bridge::StubBlock::new(1).unwrap();
            let s = block.emit(clobber as *const () as usize, 0, false).unwrap();
            block.seal().unwrap();
            s
        };
        let mut code: Vec<u8> = vec![
            0x57, // push rdi
            0x56, // push rsi
            0x48, 0x83, 0xEC, 0x28, // sub rsp, 40 (shadow space + alignment)
            0x48, 0xBF, 0x11, 0, 0, 0, 0, 0, 0, 0, // mov rdi, 0x11
            0x48, 0xBE, 0x22, 0, 0, 0, 0, 0, 0, 0, // mov rsi, 0x22
            0xB8, 0x33, 0, 0, 0, // mov eax, 0x33
            0x66, 0x48, 0x0F, 0x6E, 0xF0, // movq xmm6, rax
            0x48, 0xB8, // mov rax, stub
        ];
        code.extend((stub as u64).to_le_bytes());
        code.extend([
            0xFF, 0xD0, // call rax
            0x48, 0x8D, 0x04, 0x37, // lea rax, [rdi + rsi]
            0x66, 0x48, 0x0F, 0x7E, 0xF1, // movq rcx, xmm6
            0x48, 0x01, 0xC8, // add rax, rcx
            0x48, 0x83, 0xC4, 0x28, // add rsp, 40
            0x5E, // pop rsi
            0x5F, // pop rdi
            0xC3, // ret
        ]);
        let page = victory_platform::mem::page_size();
        // SAFETY: our own page, written then made executable.
        let f = unsafe {
            use victory_platform::mem::{self, Protect};
            let p = mem::alloc(None, page, Protect::READ_WRITE).unwrap();
            std::ptr::copy_nonoverlapping(code.as_ptr(), p, code.len());
            mem::protect(p, page, Protect::READ_EXECUTE).unwrap();
            p as usize
        };
        // SAFETY: the code above follows the Windows ABI.
        let sum = unsafe { call_win64(f, &[]) };
        assert_eq!(sum, 0x11 + 0x22 + 0x33);
    }
}
