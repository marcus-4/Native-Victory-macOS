//! macOS implementation of `sync`: mutex and condition variable objects,
//! and `os_sync` (macOS 14.4 and later, `__ulock` before) for wait-on-address.

use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits on `cv` until `ready` takes its turn (it consumes what it waits for).
fn wait_until<T>(
    mutex: &Mutex<T>,
    cv: &Condvar,
    timeout: Option<Duration>,
    mut ready: impl FnMut(&mut T) -> bool,
) -> bool {
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut state = lock(mutex);
    loop {
        if ready(&mut state) {
            return true;
        }
        state = match deadline {
            None => cv.wait(state).unwrap_or_else(|e| e.into_inner()),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                cv.wait_timeout(state, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
        };
    }
}

/// A Win32-style event: signalled or not, manual or auto reset.
#[derive(Debug)]
pub struct Event {
    manual: bool,
    state: Mutex<bool>,
    cv: Condvar,
}

impl Event {
    pub fn new(manual_reset: bool, initial: bool) -> io::Result<Event> {
        Ok(Event {
            manual: manual_reset,
            state: Mutex::new(initial),
            cv: Condvar::new(),
        })
    }

    /// Signals the event. A manual-reset event releases every waiter and
    /// stays set; an auto-reset one releases one and resets.
    pub fn set(&self) {
        *lock(&self.state) = true;
        if self.manual {
            self.cv.notify_all();
        } else {
            self.cv.notify_one();
        }
    }

    pub fn reset(&self) {
        *lock(&self.state) = false;
    }

    /// Waits until signalled; `false` if `timeout` passes first.
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let manual = self.manual;
        wait_until(&self.state, &self.cv, timeout, |set| {
            let was = *set;
            if was && !manual {
                *set = false;
            }
            was
        })
    }
}

/// A counting semaphore with a maximum, as `CreateSemaphore` makes.
#[derive(Debug)]
pub struct Semaphore {
    max: u32,
    count: Mutex<u32>,
    cv: Condvar,
}

impl Semaphore {
    pub fn new(initial: u32, max: u32) -> io::Result<Semaphore> {
        if max == 0 || initial > max {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        Ok(Semaphore {
            max,
            count: Mutex::new(initial),
            cv: Condvar::new(),
        })
    }

    /// Adds `n` to the count and returns the count before; `None` (and no
    /// change) if that would pass the maximum.
    pub fn release(&self, n: u32) -> Option<u32> {
        let mut count = lock(&self.count);
        let before = *count;
        if n > self.max - before {
            return None;
        }
        *count += n;
        drop(count);
        match n {
            0 => {}
            1 => self.cv.notify_one(),
            _ => self.cv.notify_all(),
        }
        Some(before)
    }

    /// Takes one from the count, waiting for it; `false` on timeout.
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        wait_until(&self.count, &self.cv, timeout, |count| {
            let some = *count > 0;
            if some {
                *count -= 1;
            }
            some
        })
    }
}

const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;

type OsWait = unsafe extern "C" fn(*mut u32, u64, usize, u32) -> i32;
type OsWaitTimeout = unsafe extern "C" fn(*mut u32, u64, usize, u32, u32, u64) -> i32;
type OsWake = unsafe extern "C" fn(*mut u32, usize, u32) -> i32;

/// `os_sync` exists from macOS 14.4 only, so it is looked up when first needed
/// rather than linked (a hard link fails to load, and to build, before 14.4).
/// Earlier systems get the same kernel wait through `__ulock_wait2` /
/// `__ulock_wake` (macOS 11 and later), as Rust's own std does.
struct OsSync {
    wait: OsWait,
    wait_timeout: OsWaitTimeout,
    wake_any: OsWake,
    wake_all: OsWake,
}

const RTLD_DEFAULT: *mut std::ffi::c_void = -2isize as *mut std::ffi::c_void;
const UL_COMPARE_AND_WAIT: u32 = 1;
const ULF_WAKE_ALL: u32 = 0x100;

unsafe extern "C" {
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn __ulock_wait2(op: u32, addr: *mut std::ffi::c_void, value: u64, timeout_ns: u64, value2: u64) -> i32;
    fn __ulock_wake(op: u32, addr: *mut std::ffi::c_void, wake_value: u64) -> i32;
}

fn os_sync() -> Option<&'static OsSync> {
    static FNS: std::sync::OnceLock<Option<OsSync>> = std::sync::OnceLock::new();
    FNS.get_or_init(|| {
        if std::env::var_os("VICTORY_NO_OS_SYNC").is_some() {
            return None;
        }
        let find = |name: &std::ffi::CStr| {
            // SAFETY: a lookup in the loaded images by a NUL-terminated name.
            let p = unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) };
            (!p.is_null()).then_some(p)
        };
        let wait = find(c"os_sync_wait_on_address")?;
        let wait_timeout = find(c"os_sync_wait_on_address_with_timeout")?;
        let wake_any = find(c"os_sync_wake_by_address_any")?;
        let wake_all = find(c"os_sync_wake_by_address_all")?;
        // SAFETY: these are the documented os_sync signatures.
        unsafe {
            Some(OsSync {
                wait: std::mem::transmute::<*mut std::ffi::c_void, OsWait>(wait),
                wait_timeout: std::mem::transmute::<*mut std::ffi::c_void, OsWaitTimeout>(wait_timeout),
                wake_any: std::mem::transmute::<*mut std::ffi::c_void, OsWake>(wake_any),
                wake_all: std::mem::transmute::<*mut std::ffi::c_void, OsWake>(wake_all),
            })
        }
    })
    .as_ref()
}

/// One kernel wait while `*p == expected`, for at most `timeout_ns` (`None`: no
/// limit). As `os_sync`: >= 0 when woken or the value differs, -1 with errno set.
///
/// # Safety
/// `p` is a live 4-byte aligned atomic.
unsafe fn wait_once(fns: Option<&OsSync>, p: *mut u32, expected: u32, timeout_ns: Option<u64>) -> i32 {
    // SAFETY: the caller's contract.
    unsafe {
        match (fns, timeout_ns) {
            (Some(f), None) => (f.wait)(p, u64::from(expected), 4, 0),
            (Some(f), Some(ns)) => {
                (f.wait_timeout)(p, u64::from(expected), 4, 0, OS_CLOCK_MACH_ABSOLUTE_TIME, ns)
            }
            // __ulock_wait2 takes 0 as "no limit", so a limit is at least 1 ns.
            (None, t) => __ulock_wait2(
                UL_COMPARE_AND_WAIT,
                p.cast(),
                u64::from(expected),
                t.map_or(0, |ns| ns.max(1)),
                0,
            ),
        }
    }
}

/// Wakes one or every thread waiting on `p` (no waiter is not an error).
///
/// # Safety
/// `p` is a live 4-byte aligned atomic.
unsafe fn wake(fns: Option<&OsSync>, p: *mut u32, all: bool) {
    // SAFETY: the caller's contract.
    unsafe {
        match (fns, all) {
            (Some(f), false) => (f.wake_any)(p, 4, 0),
            (Some(f), true) => (f.wake_all)(p, 4, 0),
            (None, false) => __ulock_wake(UL_COMPARE_AND_WAIT, p.cast(), 0),
            (None, true) => __ulock_wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL, p.cast(), 0),
        }
    };
}

/// Blocks while `*addr == expected` (`WaitOnAddress`). Returns at once if it
/// differs; `false` if `timeout` passes first. Spurious wake-ups are
/// possible: the caller re-checks.
pub fn wait_on_address(addr: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    if let Some(t) = timeout
        && t.is_zero()
    {
        return addr.load(Ordering::Acquire) != expected;
    }
    let deadline = timeout.map(|t| Instant::now() + t);
    let p = addr.as_ptr();
    loop {
        // SAFETY: `p` is a live 4-byte aligned atomic.
        let r = unsafe {
            match deadline {
                None => wait_once(os_sync(), p, expected, None),
                Some(d) => {
                    let Some(left) = d.checked_duration_since(Instant::now()) else {
                        return addr.load(Ordering::Acquire) != expected;
                    };
                    if left.is_zero() {
                        return addr.load(Ordering::Acquire) != expected;
                    }
                    wait_once(os_sync(), p, expected, Some(left.as_nanos().min(u128::from(u64::MAX)) as u64))
                }
            }
        };
        if r >= 0 {
            return true;
        }
        match io::Error::last_os_error().raw_os_error() {
            Some(4) => {}             // EINTR: retry with remaining time
            Some(60) => return false, // ETIMEDOUT
            _ => return true,         // EFAULT and the like: let the caller look
        }
    }
}

/// Wakes one thread blocked in [`wait_on_address`] on `addr`.
pub fn wake_one(addr: &AtomicU32) {
    // SAFETY: `addr` is a live 4-byte aligned atomic.
    unsafe { wake(os_sync(), addr.as_ptr(), false) };
}

/// Wakes every thread blocked in [`wait_on_address`] on `addr`.
pub fn wake_all(addr: &AtomicU32) {
    // SAFETY: as above.
    unsafe { wake(os_sync(), addr.as_ptr(), true) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Both kernel paths (os_sync where the system has it, and the __ulock
    /// fallback every supported macOS has): a changed value returns at once, an
    /// unchanged one times out, and a wake releases a waiter.
    #[test]
    fn wait_on_address_works_on_both_kernel_paths() {
        let paths: Vec<Option<&OsSync>> = match os_sync() {
            Some(f) => vec![Some(f), None],
            None => vec![None],
        };
        for fns in paths {
            let a = Arc::new(AtomicU32::new(7));
            let p = a.as_ptr();
            // SAFETY: `a` is live for each call.
            unsafe {
                assert!(wait_once(fns, p, 8, Some(1_000_000_000)) >= 0, "value differs");
                let t = Instant::now();
                assert_eq!(wait_once(fns, p, 7, Some(20_000_000)), -1, "timed out");
                assert_eq!(io::Error::last_os_error().raw_os_error(), Some(60));
                assert!(t.elapsed() >= Duration::from_millis(15));
                wake(fns, p, false); // no waiter
            }
            for all in [false, true] {
                a.store(7, Ordering::SeqCst);
                let w = Arc::clone(&a);
                let fns_addr = fns.map(|f| f as *const OsSync as usize);
                let h = std::thread::spawn(move || {
                    let fns = fns_addr.map(|f| unsafe { &*(f as *const OsSync) });
                    while w.load(Ordering::SeqCst) == 7 {
                        // SAFETY: `w` is live.
                        unsafe { wait_once(fns, w.as_ptr(), 7, None) };
                    }
                });
                std::thread::sleep(Duration::from_millis(20));
                a.store(9, Ordering::SeqCst);
                // SAFETY: `a` is live.
                unsafe { wake(fns, a.as_ptr(), all) };
                h.join().unwrap();
            }
        }
    }
    use super::Semaphore;
    use std::sync::{Barrier, mpsc};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn semaphore_can_release_the_maximum_count_promptly() {
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let semaphore = Semaphore::new(0, u32::MAX).unwrap();
            let previous = semaphore.release(u32::MAX);
            tx.send((previous, semaphore.release(0))).unwrap();
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2))
                .expect("bulk release timed out"),
            (Some(0), Some(u32::MAX))
        );
        worker.join().unwrap();
    }

    #[test]
    fn semaphore_waiters_can_only_take_released_permits() {
        let semaphore = Semaphore::new(0, 2).unwrap();
        let ready = Barrier::new(4);
        let (tx, rx) = mpsc::channel();
        thread::scope(|scope| {
            for _ in 0..3 {
                let (semaphore, ready, tx) = (&semaphore, &ready, tx.clone());
                scope.spawn(move || {
                    ready.wait();
                    tx.send(semaphore.wait(Some(Duration::from_secs(5))))
                        .unwrap();
                });
            }
            ready.wait();
            // Give the workers a chance to enter the condition variable wait.
            thread::sleep(Duration::from_millis(20));

            assert_eq!(semaphore.release(0), Some(0));
            assert!(!semaphore.wait(Some(Duration::ZERO)));
            assert_eq!(semaphore.release(2), Some(0));
            for _ in 0..2 {
                assert!(rx.recv_timeout(Duration::from_secs(1)).unwrap());
            }
            assert_eq!(semaphore.release(1), Some(0));
            assert!(rx.recv_timeout(Duration::from_secs(1)).unwrap());
            assert!(!semaphore.wait(Some(Duration::ZERO)));
        });
    }
}
