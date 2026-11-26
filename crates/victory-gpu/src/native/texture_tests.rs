//! Title textures on the native host: textures laid out in the console's tiled layouts in title
//! memory, detiled into host textures; textures the title places in memory itself; render targets
//! cleared through the title's views. Our own synthetic texels. Skipped without Metal.

use super::Pool;
use crate::native::glue::{self, with_list, with_title};
use crate::native::list::NativeList;
use crate::native::title::Title;
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::object::{This, slot};
use victory_backend::resources::*;
use victory_formats::texture::Swizzle;
use victory_metal::mem::{Alias, page_size};

fn pool(id: usize, pages: usize) -> Option<Pool> {
    // Needs a device; the title is made on first use.
    with_title(|_| ())?;
    let mem = Alias::new(pages * page_size().max(16384)).unwrap();
    let info = PoolInfo {
        id,
        base: mem.primary as usize,
        len: mem.len,
        memory_entry: None,
    };
    let observer = glue::pool_observer();
    observer.pool_created(&info);
    observer.mapped(mem.alias as usize, mem.len, &info, 0);
    Some(Pool { mem })
}

/// What `x`, `y` hold in mip `mip` of the test texture.
fn texel(mip: u32, x: u32, y: u32) -> [u8; 4] {
    [x as u8, y as u8, mip as u8 * 40, 255]
}

/// Tiles a texture's mips into `pool` memory at `at`, texel by `texel`.
fn tile(pool: &Pool, at: usize, layout: &victory_formats::texture::Layout) {
    let mut bytes = vec![0u8; layout.size as usize];
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let linear: Vec<u8> = (0..m.height)
            .flat_map(|y| (0..m.width).flat_map(move |x| texel(mip, x, y)))
            .collect();
        layout
            .tile(&mut bytes, mip, 0, &linear, m.width as usize * 4, linear.len())
            .unwrap();
    }
    pool.write(at, &bytes);
}

fn read_mip(key: usize, mip: usize, w: usize, h: usize) -> Vec<u8> {
    with_title(|t| {
        let tex = t.texture_of(key).expect("a host texture");
        let mut out = vec![0u8; w * h * 4];
        t.host.backend.read_texture(&tex, mip, 0, &mut out, w * 4, 0);
        out
    })
    .unwrap()
}

/// A tiled texture with a mip chain is detiled mip by mip into its host texture when work is
/// submitted, again only when its memory changed.
#[test]
fn a_tiled_title_texture_is_detiled_into_the_host_texture() {
    let Some(pool) = pool(9101, 8) else { return };
    // The XOR modes (`_X`, `_T`) too: their equations come from the pipe configuration, not a
    // fixed pattern, and a texture of theirs is no longer left empty.
    for swizzle in [
        Swizzle::LINEAR,
        Swizzle::SW_256B_S,
        Swizzle::SW_4KB_S,
        Swizzle::SW_64KB_D,
        Swizzle::SW_4KB_S_X,
        Swizzle::SW_64KB_D_X,
        Swizzle::SW_64KB_S_T,
    ] {
        let at = 0;
        let res = crate::resource::test_texture_at(pool.va(at), (64, 32), 4, 28, swizzle.0, 0);
        let info = crate::resource::texture_info(res).unwrap();
        tile(&pool, at, &info.layout);
        let key = res as usize;
        let uploads = |key| with_title(|t| t.uploads_of(key)).unwrap();
        with_title(|t| t.sync_textures());
        // Nothing named the texture yet: no host texture, nothing detiled.
        assert_eq!(uploads(key), 0);
        let mips: Vec<(usize, usize)> = (0..4).map(|m| (64 >> m, 32 >> m)).collect();
        // Naming it makes it; the next sync fills it.
        with_title(|t| {
            assert!(t.realize(key));
            t.sync_textures();
        });
        for (mip, &(w, h)) in mips.iter().enumerate() {
            let out = read_mip(key, mip, w, h);
            for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w / 2, h / 2)] {
                let at = (y * w + x) * 4;
                assert_eq!(
                    out[at..at + 4],
                    texel(mip as u32, x as u32, y as u32),
                    "{swizzle:?} mip {mip} at {x},{y}"
                );
            }
        }
        assert_eq!(uploads(key), 1, "uploaded once");
        // Unchanged memory is not detiled again; changed memory is.
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 1);
        pool.write(at, &[0xFFu8; 16]);
        with_title(|t| t.sync_textures());
        assert_eq!(uploads(key), 2);
    }
}

/// Held by the tests that count on a texture staying hot: the many submissions of
/// `a_texture_blank_for_thousands_of_submissions_is_still_probed` cool every other test's textures.
static AGING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A texture uploaded blank and written whole later (a movie's planes) is uploaded again at the
/// next submission, not a scanner round later (a round is seconds long once the title streams).
#[test]
fn a_texture_uploaded_blank_is_probed_at_each_submission() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    // 256 KiB: 64 pages, more than a sampling probe reads.
    let Some(pool) = pool(9105, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1, "the blank first upload");
    });
    pool.write(0, &vec![0xABu8; info.size]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 2, "uploaded again at the next submission");
    });
    // Its samples showed that write, as they show a streamed texture's: it is not read whole at
    // every submission after it (thousands of those were most of what a world submission read).
    with_title(|t| {
        let reads = (0..crate::native::textures::WHOLE_EVERY)
            .filter(|_| {
                let jobs = t.sync_jobs();
                let read = jobs.iter().any(|j| j.key == key);
                t.sync(jobs);
                read
            })
            .count();
        assert!(reads <= 1, "read whole {reads} times in a whole-read period");
    });
    // A glyph's worth of bytes in page 3 (a page a sampling probe would skip).
    pool.write(3 * 4096 + 100, &[1u8; 16]);
    // (Twice the period: other tests' textures share what a submission spares for these reads.)
    with_title(|t| {
        let n = (1..=2 * crate::native::textures::WHOLE_EVERY).find(|_| {
            t.sync_dirty();
            t.uploads_of(key) == 3
        });
        assert!(n.is_some(), "a small write is seen within a whole-read period or two");
    });
    pool.write(5 * 4096 + 100, &[1u8; 16]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 4, "and, its samples having missed one, the next at the next submission");
    });
}

/// A texture uploaded blank stays hot however long it stays blank: the boot movie's planes are
/// made about 40 s (thousands of submissions) before the movie decodes into them, and once cold
/// their first frames waited seconds for the scanner (green).
#[test]
fn a_texture_blank_for_thousands_of_submissions_is_still_probed() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9123, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        for _ in 0..2000 {
            t.sync_dirty();
        }
        assert_eq!(t.uploads_of(key), 1, "the blank first upload");
    });
    pool.write(0, &vec![0xABu8; info.size]);
    // Within a few submissions: other tests' blank textures share the probe's time budget.
    with_title(|t| {
        let n = (1..=4).find(|_| {
            t.sync_dirty();
            t.uploads_of(key) == 2
        });
        assert!(n.is_some(), "the first frame is uploaded within a few submissions");
    });
}

/// A texture rewritten whole (a movie's chroma plane) is not read whole at every submission, yet
/// a small write its samples miss is still seen within `WHOLE_EVERY` submissions, and from then
/// on at the next one (a glyph atlas).
#[test]
fn a_texture_rewritten_whole_is_read_whole_only_now_and_then() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9124, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    let sync = || with_title(|t| { t.sync_dirty(); t.uploads_of(key) }).unwrap();
    with_title(|t| assert!(t.realize(key)));
    assert_eq!(sync(), 1, "the blank first upload");
    for (i, n) in [(1u8, 2), (2, 3), (3, 4)] {
        pool.write(0, &vec![i; info.size]);
        assert_eq!(sync(), n, "a whole rewrite is seen by the samples at once");
        sync();
    }
    // A glyph's worth of bytes in page 3 (a page the samples skip).
    pool.write(3 * 4096 + 100, &[9u8; 16]);
    // (Twice the period: other tests' textures share what a submission spares for these reads.)
    let n = (1..=2 * crate::native::textures::WHOLE_EVERY).find(|_| sync() == 5);
    assert!(n.is_some(), "a write the samples miss is seen within a whole-read period or two");
    pool.write(5 * 4096 + 100, &[9u8; 16]);
    assert_eq!(sync(), 6, "and the next one at the next submission");
}

/// A texture whose last write missed the sampled pages (a streamed mip tail, or a single glyph) is
/// checked whole at the next probe for a burst, then returns to periodic checks with backoff once
/// a whole read finds nothing changed instead of reading whole at every probe until it cools.
#[test]
fn a_missed_write_not_followed_by_another_backs_off_whole_reads() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = pool(9131, (256 << 10) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (256, 256), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1);
    });
    pool.write(0, &vec![0xABu8; info.size]);
    with_title(|t| {
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 2);
    });
    // Write in page 3, which the 8 sampled pages skip.
    pool.write(3 * 4096 + 100, &[7u8; 16]);
    with_title(|t| {
        let found = (1..=2 * crate::native::textures::WHOLE_EVERY).any(|_| {
            t.sync_dirty();
            t.uploads_of(key) == 3
        });
        assert!(found, "missed-sample write uploaded");
    });
    // Over 64 idle submissions: 1 immediate follow-up check, then +8, +16, +32 (at most 4 reads,
    // instead of all 64).
    with_title(|t| {
        let reads = (0..64)
            .filter(|_| {
                let jobs = t.sync_jobs();
                let read = jobs.iter().any(|j| j.key == key);
                t.sync(jobs);
                read
            })
            .count();
        assert!(reads <= 4, "unchanged texture read whole {reads} times across 64 idle submissions");
    });
}

/// A movie plane uploaded blank among hundreds of other hot textures (a title streaming) is still
/// probed at the next submission when the probe deadline allows it: a count limit on the probes
/// left its first frames green.
/// (Every texture is written, so which one a limited probe reaches first does not matter.)
#[test]
fn a_hot_texture_is_probed_at_each_submission_among_hundreds() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    const N: usize = 200;
    const STRIDE: usize = 64 << 10;
    let Some(pool) = pool(9121, N * STRIDE / page_size() + 1) else { return };
    let keys: Vec<usize> = (0..N)
        .map(|i| crate::resource::test_texture_at(pool.va(i * STRIDE), (64, 64), 1, 28, Swizzle::LINEAR.0, 0) as usize)
        .collect();
    let size = crate::resource::texture_info(keys[0] as _).unwrap().size;
    with_title(|t| {
        for &k in &keys {
            assert!(t.realize(k));
        }
        t.sync_dirty();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 1), "every blank upload");
        let bytes = vec![0xABu8; size];
        for i in 0..N {
            pool.write(i * STRIDE, &bytes);
        }
        t.sync_dirty_without_probe_deadline();
        // The scanner can trigger uploads too. The probe removes written textures from blank_keys.
        let probed = keys.iter().filter(|&k| !t.blank_keys.contains(k)).count();
        assert_eq!(probed, N, "all {N} blank textures probed in one submission");
        let seen = keys.iter().filter(|&&k| t.uploads_of(k) == 2).count();
        assert_eq!(seen, N, "all {N} hot textures written, seen at the next submission");
    });
}

/// A submission's textures are read and detiled on several threads: each host texture gets its
/// own memory's texels, once.
#[test]
fn textures_read_on_several_threads_get_their_own_texels() {
    const N: usize = 48;
    const STRIDE: usize = 64 << 10;
    let Some(pool) = pool(9125, N * STRIDE / page_size() + 1) else { return };
    let keys: Vec<usize> = (0..N)
        .map(|i| {
            let res = crate::resource::test_texture_at(pool.va(i * STRIDE), (64, 64), 1, 28, Swizzle::SW_4KB_S.0, 0);
            let layout = crate::resource::texture_info(res).unwrap().layout;
            let mut bytes = vec![0u8; layout.size as usize];
            let texels = vec![i as u8 + 1; 64 * 64 * 4];
            layout.tile(&mut bytes, 0, 0, &texels, 64 * 4, texels.len()).unwrap();
            pool.write(i * STRIDE, &bytes);
            res as usize
        })
        .collect();
    with_title(|t| {
        for &k in &keys {
            assert!(t.realize(k));
        }
        t.sync_dirty();
        assert!(keys.iter().all(|&k| t.uploads_of(k) == 1), "each uploaded once");
    });
    for (i, &k) in keys.iter().enumerate() {
        assert!(read_mip(k, 0, 64, 64).iter().all(|&b| b == i as u8 + 1), "texture {i} holds its own texels");
    }
}

/// A texture released while a submission read its memory (without the host's lock): what was read
/// is not uploaded into the texture that has its key by then.
#[test]
fn memory_read_for_a_texture_released_meanwhile_is_dropped() {
    let Some(pool) = pool(9126, 8) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    pool.write(0, &vec![7u8; info.size]);
    with_title(|t| {
        assert!(t.realize(key));
        let jobs = t.sync_jobs();
        assert!(jobs.iter().any(|j| j.key == key));
        let prepared = crate::native::textures::prepare(&jobs);
        // Released, and another made at its key, while the memory was read.
        t.texture_released(key);
        t.texture_created(key, info.clone());
        assert!(t.realize(key));
        t.sync_apply(prepared);
        assert_eq!(t.uploads_of(key), 0, "the new texture gets nothing read for the old one");
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1, "and is filled at the next submission");
    });
}

/// A GPU copy into a texture recorded while a submission read its memory (without the host's
/// lock): the texture is the GPU's by then, and what was read is not uploaded over it.
#[test]
fn memory_read_for_a_texture_a_copy_writes_meanwhile_is_dropped() {
    let Some(pool) = pool(9128, 8) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    pool.write(0, &vec![7u8; info.size]);
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_dirty();
        assert_eq!(t.uploads_of(key), 1);
        pool.write(0, &vec![8u8; info.size]);
        let jobs = t.jobs(vec![key]);
        assert_eq!(jobs.len(), 1);
        let prepared = crate::native::textures::prepare(&jobs);
        t.gpu_written(key);
        t.sync_apply(prepared);
        assert_eq!(t.uploads_of(key), 1, "the copy's destination is not filled from title memory");
        assert!(t.jobs(vec![key]).is_empty(), "nor read for it again");
    });
}

/// A texture just uploaded is not read whole again until its memory changes: neither its first
/// probe as a hot texture nor the scanner's next round reports what the submission read.
#[test]
fn a_texture_just_uploaded_is_not_read_again_until_it_changes() {
    let _aging = AGING.lock().unwrap_or_else(|e| e.into_inner());
    // 8 MiB: more than a probe reads whole.
    let Some(pool) = pool(9127, (8 << 20) / page_size() + 1) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (2048, 1024), 1, 28, Swizzle::LINEAR.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    let key = res as usize;
    // A submission: whether it reads the texture, and the texture's uploads after it.
    let submit = || {
        with_title(|t| {
            let jobs = t.sync_jobs();
            let read = jobs.iter().any(|j| j.key == key);
            t.sync(jobs);
            (read, t.uploads_of(key))
        })
        .unwrap()
    };
    // Written under the lock, so that no other test's submission reads half of it.
    let write = |bytes: &[u8]| with_title(|_| pool.write(0, bytes)).unwrap();
    with_title(|t| assert!(t.realize(key)));
    assert_eq!(submit().1, 1, "the blank first upload");
    write(&vec![0xABu8; info.size]);
    // Within a few submissions (other tests' blank textures share the probe's time budget), this
    // test's or another's.
    assert!((1..=4).any(|_| submit().1 == 2), "written, so uploaded again");
    for _ in 0..20 {
        // Time for a round of the scanner.
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(submit(), (false, 2), "read again though nothing was written");
    }
    write(&[1u8; 16]);
    assert!((1..=4).any(|_| submit().1 == 3), "a write is seen");
}

/// A block-compressed texture: 16 bytes a 4x4 block, in the console's layout, detiled as blocks.
#[test]
fn a_block_compressed_title_texture_is_detiled_as_blocks() {
    let Some(pool) = pool(9102, 4) else { return };
    // BC7 (98): a 32x16 texture is 8x4 blocks.
    let res = crate::resource::test_texture_at(pool.va(0), (32, 16), 1, 98, Swizzle::SW_64KB_D.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    assert_eq!(info.layout.desc.element_bytes, 16);
    let m = &info.layout.mips[0];
    let blocks: Vec<u8> = (0..m.width * m.height * 16).map(|i| (i * 7) as u8).collect();
    let mut bytes = vec![0u8; info.layout.size as usize];
    info.layout
        .tile(&mut bytes, 0, 0, &blocks, m.width as usize * 16, blocks.len())
        .unwrap();
    pool.write(0, &bytes);
    let key = res as usize;
    with_title(|t| {
        assert!(t.realize(key));
        t.sync_textures();
        let tex = t.texture_of(key).unwrap();
        let mut out = vec![0u8; blocks.len()];
        t.host.backend.read_texture(&tex, 0, 0, &mut out, m.width as usize * 16, 0);
        assert_eq!(out, blocks);
    });
}

/// A texture a view names by memory address and description, without a resource object (the
/// console's `CreateShaderResourceViewPlaced`): the host makes it when the view is written.
#[test]
fn a_placed_texture_view_makes_and_fills_its_texture() {
    let Some(pool) = pool(9103, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (32, 32), 1, 28, Swizzle::SW_64KB_S.0, 0);
    let info = crate::resource::texture_info(res).unwrap();
    tile(&pool, 0, &info.layout);
    let words = crate::resource::texture_desc(res).unwrap();
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let mut view = [0u32; 16];
    (view[0], view[1], view[5]) = (28, 4, 1);
    (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
    // The heap is made (and the view written into it) when a list names it.
    let key = crate::native::textures::placed_key(pool.va(0), &words);
    with_title(|t| {
        assert!(t.host.knows(key), "made when the view was written");
        t.sync_textures();
    });
    let out = read_mip(key, 0, 32, 32);
    assert_eq!(out[(5 * 32 + 9) * 4..][..4], texel(0, 9, 5));
}

/// A placed texture has no object the title releases: it goes when another description is placed
/// at its address, and when its pool is released.
#[test]
fn a_placed_texture_goes_when_replaced_and_with_its_pool() {
    let Some(pool) = pool(9107, 4) else { return };
    let heap = crate::descriptor::test_heap(false, 4);
    let start = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let place = |size: u32| {
        let res = crate::resource::test_texture_at(pool.va(0), (size.into(), size), 1, 28, Swizzle::SW_64KB_S.0, 0);
        let words = crate::resource::texture_desc(res).unwrap();
        let mut view = [0u32; 16];
        (view[0], view[1], view[5]) = (28, 4, 1);
        (view[10], view[11]) = (pool.va(0) as u32, (pool.va(0) >> 32) as u32);
        crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, start);
        crate::native::textures::placed_key(pool.va(0), &words)
    };
    let first = place(32);
    assert!(with_title(|t| t.host.knows(first)).unwrap());
    let second = place(16);
    assert_ne!(first, second);
    with_title(|t| {
        assert!(t.host.knows(second));
        assert!(!t.host.knows(first), "replaced at its address");
    });
    let info = PoolInfo {
        id: 9107,
        base: pool.mem.primary as usize,
        len: pool.mem.len,
        memory_entry: None,
    };
    glue::pool_observer().pool_released(&info);
    with_title(|t| {
        t.sync_dirty();
        assert!(!t.host.knows(second), "released with its pool");
        assert!(t.texture_of(second).is_none());
    });
}

/// A render target the title clears through its view: the host texture is the target, never
/// filled from memory.
#[test]
fn a_render_target_is_cleared_through_the_titles_view() {
    let Some(pool) = pool(9104, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    pool.write(0, &[0x7Fu8; 64]);
    // An RTV in descriptor memory the title owns.
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let list = crate::list::test_list(false, std::ptr::null());
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    // OMSetRenderTargets(1, &handle, false, null), then ClearRenderTargetView(handle, color).
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    let color = [0.0f32, 1.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
            f,
            1,
            std::ptr::null_mut()
        ),
        0
    );
    let out = read_mip(rt as usize, 0, 16, 16);
    assert!(out.chunks(4).all(|p| p == [0, 255, 0, 255]), "{:?}", &out[..8]);
}

/// A render target view names the format its writes are stored in. Through a plain view of an
/// sRGB texture a value is stored as it is (a movie's picture, which is display values already);
/// through an sRGB view of a plain texture it is encoded.
#[test]
fn a_render_target_view_stores_in_the_format_it_names() {
    let Some(pool) = pool(9120, 16) else { return };
    // The texture's format, the view's (0: none described), and the byte half grey is stored as.
    for (i, (own, named, stored)) in [(29, 28, 128), (28, 29, 188), (29, 0, 188), (28, 28, 128)]
        .into_iter()
        .enumerate()
    {
        let at = i * 0x10000;
        let rt = crate::resource::test_texture_at(
            pool.va(at),
            (16, 16),
            1,
            own,
            Swizzle::SW_64KB_R_X.0,
            1,
        );
        let heap = crate::descriptor::test_heap(false, 4);
        let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
        // A 2D view of mip 0.
        let desc = [named, 4, 0, 0, 0, 0];
        let desc = if named == 0 {
            std::ptr::null()
        } else {
            &raw const desc
        };
        crate::descriptor::create_rtv(std::ptr::null(), rt, desc, handle);
        let list = crate::list::test_list(false, std::ptr::null());
        let (q, f) = (crate::queue::test_queue(false), {
            let mut f = std::ptr::null();
            crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
            f
        });
        slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
            list,
            1,
            &handle,
            0,
            std::ptr::null(),
        );
        let color = [0.5f32, 0.5, 0.5, 1.0];
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
            list,
            handle,
            &color,
            0,
            std::ptr::null(),
        );
        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
        slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
        assert_eq!(
            slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
                f,
                1,
                std::ptr::null_mut()
            ),
            0
        );
        let out = read_mip(rt as usize, 0, 16, 16);
        assert!(
            out.chunks(4)
                .all(|p| p[0].abs_diff(stored) <= 1 && p[3] == 255),
            "texture {own} view {named}: {:?}",
            &out[..4]
        );
    }
}

/// A layout the host cannot address (`SW_4KB_R_X`, which has no equations) is counted and left empty, not
/// an error that stops the submission.
#[test]
fn an_undetilable_layout_is_counted_not_fatal() {
    let Some(pool) = pool(9105, 4) else { return };
    let res = crate::resource::test_texture_at(pool.va(0), (64, 64), 1, 28, Swizzle::SW_4KB_R_X.0, 0);
    let before = with_title(|t| t.stats.undetilable).unwrap();
    with_title(|t| {
        assert!(t.realize(res as usize));
        t.sync_textures();
        t.sync_textures();
    });
    assert_eq!(with_title(|t| t.stats.undetilable).unwrap(), before + 1, "once per content");
}

/// `PresentX`: the first resource of the largest plane is drawn over the app's layer (scaled,
/// whatever its channel order), after the queue's earlier work. Offscreen: the layer is not in a
/// window; its last drawable is read back.
#[test]
fn present_draws_the_back_buffer_over_the_apps_layer() {
    check_present_order(false);
    check_present_order(true);
}

fn check_present_order(held: bool) {
    let Some(pool) = pool(if held { 9116 } else { 9106 }, 8) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_S.0, 1);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    let layer = glue::create_layer(32, 32).expect("a layer");
    glue::set_layer(layer.as_ptr()).expect("a surface");
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    let mut gate = std::ptr::null();
    crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut gate);
    if held {
        glue::queue_wait(q, gate, 1);
    }
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    // Red in the title's RGBA.
    let color = [1.0f32, 0.0, 0.0, 1.0];
    slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0)(
        list,
        handle,
        &color,
        0,
        std::ptr::null(),
    );
    slot::<extern "system" fn(This) -> i32>(list, 0x48)(list);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    // One plane: the frame token, one resource, its address.
    #[repr(C)]
    struct Plane {
        token: u64,
        count: u32,
        pad: u32,
        resources: *const This,
    }
    let plane = Plane {
        token: 1,
        count: 1,
        pad: 0,
        resources: &rt,
    };
    let before = glue::frames_presented();
    let present = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> i32>(q, 0xf0);
    let frames = if held { 6 } else { 1 };
    for _ in 0..frames {
        assert_eq!(present(q, 1, &plane, std::ptr::null()), 0);
    }
    // A signal after the present must also stay behind it.
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    if held {
        // Present must remain behind the unsignalled wait and the red clear.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(glue::frames_presented(), before, "present bypassed the fence wait");
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 0);
        // Signal the backend directly so only the timer worker drains the deferred presents.
        glue::with_title(|t| {
            use victory_backend::backend::ComputeBackend;
            let f = t.fence(gate as usize, 0).unwrap();
            t.host.backend.signal(&f, 1);
        });
    }
    let start = std::time::Instant::now();
    while glue::frames_presented() < before + frames {
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "worker failed to drain presents");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(glue::frames_presented(), before + frames);
    // The frame is drawn when the fence is reached.
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()),
        0
    );
    let (w, h, bgra) = glue::last_frame().expect("a frame");
    assert_eq!((w, h), (32, 32));
    assert!(
        bgra.chunks(4).all(|p| p == [0, 0, 255, 255]),
        "red as BGRA: {:?}",
        &bgra[..8]
    );
    drop(layer);
}

/// A colour target the host cannot make must keep its slot: binding the later ones would put
/// `SV_Target1` on slot 0 and write a surface this pass does not own. Nothing is bound instead,
/// and the draws that follow are refused.
#[test]
fn a_target_the_host_cannot_make_does_not_shift_the_ones_after_it() {
    let Some(pool) = pool(9201, 4) else { return };
    // 12 (R16G16B16A16_UINT) has no host texture; 28 is R8G8B8A8_UNORM.
    let gone = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 12, Swizzle::SW_64KB_R_X.0, 1);
    let good = crate::resource::test_texture_at(pool.va(0x2000), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    // Two view handles, 64 bytes apart, as a title's own descriptor memory holds them.
    let mut slots = vec![0u64; 16];
    let first = slots.as_mut_ptr() as u64;
    let second = first + 64;
    crate::descriptor::create_rtv(std::ptr::null(), gone, std::ptr::null(), first);
    crate::descriptor::create_rtv(std::ptr::null(), good, std::ptr::null(), second);
    let list = crate::list::test_list(false, std::ptr::null());
    let handles = [first, second];
    // OMSetRenderTargets(2, &handles, false, null).
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        handles.len() as u32,
        handles.as_ptr(),
        0,
        std::ptr::null(),
    );
    let bound = with_list(list, |_, l| l.targets.color.len()).unwrap();
    assert_eq!(bound, 0, "a target without a host texture must not renumber the ones after it");
    // A set the host can make is taken again.
    let only = [second];
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        only.len() as u32,
        only.as_ptr(),
        0,
        std::ptr::null(),
    );
    let bound = with_list(list, |_, l| l.targets.color.len()).unwrap();
    assert_eq!(bound, 1, "a makeable target binds again after a refused pass");
    drop(slots);
}

/// A compute dispatch writes a tiled title texture through a UAV; once its fence value shows, the
/// title's own memory holds the texels, in the texture's layout (the ocean maps' path, #46).
#[test]
fn a_texture_compute_writes_is_copied_back_into_title_memory() {
    // Skips without the Shader Converter.
    if super::host().is_none() {
        return;
    }
    let dxil = include_bytes!("../../../victory-metal/tests/fixtures/tex_fill.dxil");
    const W: u32 = 64;
    // A title host of its own: no timer thread, and a pool no other test shares.
    let mut t = Title::new(victory_metal::metal::Metal::new().unwrap(), Default::default());
    let mem = Alias::new((64 << 10).max(page_size())).unwrap();
    let info = PoolInfo { id: 9131, base: mem.primary as usize, len: mem.len, memory_entry: None };
    t.host.memory().pool_created(&info);
    t.host.memory().mapped(mem.alias as usize, mem.len, &info, 0);
    let pool = Pool { mem };
    t.set_texture_back(true);
    for (n, swizzle) in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_D_X].into_iter().enumerate() {
        // Flags 4: unordered access. DXGI 28: RGBA8.
        let res = crate::resource::test_texture_at(pool.va(0), (u64::from(W), W), 1, 28, swizzle.0, 4);
        let info = crate::resource::texture_info(res).unwrap();
        let key = res as usize;
        pool.write(0, &vec![0u8; info.size]);
        // The title's descriptor memory: one UAV (format, 2D, mip 0).
        let slots = vec![0u32; 8 * 8];
        let handle = slots.as_ptr() as u64;
        let uav = [28u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        crate::descriptor::create_uav(std::ptr::null(), res, std::ptr::null(), &uav, handle);
        let (queue, fence) = (1 + n, 11 + n);
        t.texture_created(key, info.clone());
        assert!(t.realize(key), "{swizzle:?}");
        let heap = t.host.heap(HeapKind::Resource, handle, 8).unwrap();
        let pso = t.host.compute_pipeline(dxil, [8, 8, 1]).unwrap();
        let mut l = NativeList::default();
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap, 0);
        l.dispatch(W / 8, W / 8, 1).unwrap();
        t.ensure_queue(queue).unwrap();
        t.submit(queue, vec![(l, vec![])]);
        t.signal_queue(queue, fence, 0, 1);
        let start = std::time::Instant::now();
        while !t.try_wait(fence, 0, 1) {
            assert!(start.elapsed().as_secs() < 10, "the dispatch did not complete");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // The value shows, so the memory is there already.
        let mut out = vec![0u8; info.size];
        assert!(crate::native::safe_read(pool.va(0) as *const u8, &mut out));
        let linear: Vec<u8> = (0..W)
            .flat_map(|y| (0..W).flat_map(move |x| [x as u8, y as u8, 64, 255]))
            .collect();
        let mut want = vec![0u8; info.size];
        info.layout
            .tile(&mut want, 0, 0, &linear, W as usize * 4, linear.len())
            .unwrap();
        assert!(out == want, "{swizzle:?}: title memory is not the retiled texels");
        // Our own write is not mistaken for the title's: nothing is uploaded again.
        let uploads = t.uploads_of(key);
        t.sync_textures();
        assert_eq!(t.uploads_of(key), uploads, "{swizzle:?}");
        // The title writes the memory while the same work runs: its bytes stay.
        let mut l = NativeList::default();
        let heap = t.host.heap_at(handle).unwrap().0;
        l.set_heaps(Some(&heap), None);
        l.set_compute_pipeline(&pso);
        l.set_compute_root_table(0, &heap, 0);
        l.dispatch(W / 8, W / 8, 1).unwrap();
        t.submit(queue, vec![(l, vec![])]);
        t.signal_queue(queue, fence, 0, 2);
        pool.write(0, &vec![0x5Au8; info.size]);
        let start = std::time::Instant::now();
        while !t.try_wait(fence, 0, 2) {
            assert!(start.elapsed().as_secs() < 10, "the dispatch did not complete");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(crate::native::safe_read(pool.va(0) as *const u8, &mut out));
        assert!(out.iter().all(|&b| b == 0x5A), "{swizzle:?}: the title's own write was overwritten");
        t.host.drop_heap(handle);
        t.host.forget_resource(key);
        t.texture_released(key);
    }
}

/// Depth and depth-stencil textures are GPU targets whose CPU tile layout (e.g. a 4-byte or
/// 2-byte depth plane) does not match Metal's combined texel layout, so they must not be watched
/// or uploaded from CPU memory even when placed/typeless (not marked as a render target).
#[test]
fn depth_textures_are_not_watched_or_uploaded_from_cpu_memory() {
    let Some(pool) = pool(9202, (512 << 10) / page_size() + 1) else { return };
    // DXGI formats: 20 (D32_FLOAT_S8X24_UINT), 40 (D32_FLOAT), 55 (D16_UNORM),
    // and typeless 19 (R32G8X24_TYPELESS) with flags = 0 (not marked ALLOW_RENDER_TARGET).
    for (i, format) in [20u32, 40, 55, 19].into_iter().enumerate() {
        let at = i * 0x20000;
        let res = crate::resource::test_texture_at(
            pool.va(at),
            (64, 64),
            1,
            format,
            Swizzle::SW_64KB_R_X.0,
            0,
        );
        let key = res as usize;
        with_title(|t| {
            assert!(t.realize(key), "format {format} should realize");
            t.sync_textures();
            assert_eq!(t.uploads_of(key), 0, "depth format {format} must not upload on sync_textures");
        });
        pool.write(at, &[0xA5u8; 64]);
        with_title(|t| {
            t.sync_dirty();
            assert_eq!(t.uploads_of(key), 0, "depth format {format} must not upload on sync_dirty");
        });
    }
}

/// A described RTV binds and clears its selected subresource in the named format.
#[test]
fn a_render_target_view_selects_its_mip_and_slice() {
    use std::rc::Rc;
    use victory_backend::backend::ComputeBackend;

    let pool = pool(9138, 16).expect("Metal device and generated texture pool");
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let other_heap = crate::descriptor::test_heap(false, 4);
    let next_handle = slot::<extern "system" fn(This) -> u64>(other_heap, 0x48)(other_heap);
    let grey = [0.5, 0.5, 0.5, 1.0];

    for (case, own_format, layers, array_view, mip, slice) in [
        ("2D mip", 28, 1, false, 1, 0),
        ("array slice", 28, 2, true, 0, 1),
        ("array mip and slice", 28, 2, true, 1, 1),
        ("plain view of an sRGB array mip and slice", 29, 2, true, 1, 1),
        ("nonarray view of the first slice", 28, 2, false, 1, 0),
    ] {
        let rt = crate::resource::test_texture_array_at(pool.va(0), (8, 8), 2, layers, own_format, Swizzle::LINEAR.0, 1);
        let key = rt as usize;
        let subresources: Vec<_> = (0..usize::from(layers))
            .flat_map(|layer| (0..2).map(move |level| {
                (level, layer, 8 >> level, [17 + (level * 40 + layer * 80) as u8, 31, 47, 255])
            }))
            .collect();
        with_title(|title| {
            let texture = title.texture_of(key).expect("realized texture");
            for &(level, layer, n, sentinel) in &subresources {
                title.host.backend.upload_texture(&texture, level, layer, &sentinel.repeat(n * n), n * 4, 0);
            }
        }).expect("Metal title");
        let desc = if array_view {
            [28, 5, mip, slice, 1, 0]
        } else {
            [28, 4, mip, 0, 0, 0]
        };
        crate::descriptor::create_rtv(std::ptr::null(), rt, &desc, handle);
        crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), next_handle);
        let list = crate::list::test_list(false, std::ptr::null());
        glue::targets(list, &[handle], 0);
        with_list(list, |title, l| {
            let (expected, format) = title.target_of(crate::descriptor::rtv_record(handle).unwrap()).unwrap();
            let (base, base_format) = title.target_of(crate::descriptor::rtv_record(next_handle).unwrap()).unwrap();
            assert!(l.targets_ready);
            assert_eq!(l.targets.color, [Format::Rgba8Unorm]);
            assert_eq!(format, Format::Rgba8Unorm);
            let [attached] = l.list.color_targets() else { panic!("one attachment was expected") };
            assert!(Rc::ptr_eq(attached, &expected), "binding reuses the cached attachment");
            assert!(Rc::ptr_eq(&base, &title.texture_of(key).unwrap()), "default keeps base attachment");
            assert!(!Rc::ptr_eq(&base, attached));
            assert_eq!(base_format, if own_format == 29 { Format::Rgba8UnormSrgb } else { Format::Rgba8Unorm });
            if array_view {
                for (other_mip, other_slice) in [(1 - mip, slice), (mip, 1 - slice)] {
                    let other_desc = [28, 5, other_mip, other_slice, 1, 0];
                    crate::descriptor::create_rtv(std::ptr::null(), rt, &other_desc, next_handle);
                    let (other, _) = title.target_of(crate::descriptor::rtv_record(next_handle).unwrap()).unwrap();
                    assert!(!Rc::ptr_eq(attached, &other), "distinct subresources must keep distinct identities");
                }
            }
        }).expect("native command list");

        let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const u8)>(list, 0x1a0);
        clear(list, handle, &grey, 0, std::ptr::null());
        with_list(list, |_, l| {
            let [super::super::list::Cmd::Clear { color: Some((cleared, _)), .. }] = l.list.cmds.as_slice() else {
                panic!("one clear was expected");
            };
            assert!(Rc::ptr_eq(&l.list.color_targets()[0], cleared), "draw binding and clear resolve the same view");
        }).expect("native command list");
        if layers == 2 {
            for invalid in [
                [28, 5, 2, 0, 1, 0], // Out-of-range mip.
                [28, 5, 0, 2, 1, 0], // Out-of-range slice.
                [28, 5, 0, 0, 0, 0], // Empty array view.
                [28, 5, 0, 0, 2, 0], // Unsupported layered view.
                [28, 5, 0, 0, 1, 1], // Unsupported plane.
                [28, 2, 0, 0, 0, 0], // 1D view of a 2D resource.
                [28, 6, 0, 0, 0, 0], // Multisampled view of a single-sampled resource.
            ] {
                crate::descriptor::create_rtv(std::ptr::null(), rt, &invalid, next_handle);
                glue::targets(list, &[next_handle, handle], 0);
                clear(list, next_handle, &[1.0, 0.0, 0.0, 1.0], 0, std::ptr::null());
                with_list(list, |_, l| {
                    assert!(!l.targets_ready, "{case} accepted {invalid:?}");
                    assert!(l.list.color_targets().is_empty(), "valid attachment must keep its slot");
                    assert_eq!(l.list.cmds.len(), 1, "an invalid clear must record no command");
                }).expect("native command list");
            }
        }

        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        let queue = crate::queue::test_queue(false);
        let mut fence = std::ptr::null();
        assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), 0);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, 1), 0);
        with_title(|title| {
            let fence = title.fence(fence as usize, 0).expect("native fence");
            assert!(title.host.backend.wait(&fence, 1, 2000), "native clear did not finish");
            let texture = title.texture_of(key).expect("realized texture");
            let mut pixels = 0;
            for &(level, layer, n, sentinel) in &subresources {
                let expected = if (level, layer) == (mip as usize, slice as usize) { [128, 128, 128, 255] } else { sentinel };
                let mut data = vec![0u8; n * n * 4];
                title.host.backend.read_texture(&texture, level, layer, &mut data, n * 4, 0);
                assert!(data.as_chunks::<4>().0.iter().all(|p| *p == expected),
                        "{case} mip {level} slice {layer}: expected {expected:?}, got {:?}", &data[..4]);
                pixels += n * n;
            }
            println!("{case}: cleared mip {mip} slice {slice}, verified {pixels} texels");
        }).expect("Metal title");
    }
}

/// A rectangle-list draw (`D3D_PRIMITIVE_TOPOLOGY_RECTLIST`, 17) on a described (non-guessed)
/// pipeline must draw the supplied vertices as `Topology::Triangles` instead of issuing
/// 4-vertex instanced strips designed for `Options::rect_list`.
#[test]
fn described_rectangle_list_draw_uses_supplied_triangles() {
    let Some(pool) = pool(9204, 4) else { return };
    let rt = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::SW_64KB_R_X.0, 1);
    let pso = with_title(|t| {
        let vs = super::fixture("tri_vs");
        let ps = super::fixture("tri_ps");
        t.host
            .render_pipeline(&RenderPipelineDesc {
                vs: &vs,
                ps: Some(&ps),
                vs_layout: None,
                ps_layout: None,
                attributes: &[
                    VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 0 },
                    VertexAttr { format: VertexFormat::Float2, slot: 0, offset: 8 },
                    VertexAttr { format: VertexFormat::Float4, slot: 0, offset: 16 },
                ],
                streams: &[VertexStream { slot: 0, stride: 32, per_instance: false }],
                color: &[Format::Rgba8Unorm],
                depth: None,
                depth_test: None,
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap()
    })
    .unwrap();
    let pipe_key = 0xDEAD_BEE0usize;
    glue::install_test_render_pipeline(pipe_key, pso);

    let list = crate::list::test_list(false, std::ptr::null());
    glue::set_pipeline(list, pipe_key as This);
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    crate::descriptor::create_rtv(std::ptr::null(), rt, std::ptr::null(), handle);
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &handle,
        0,
        std::ptr::null(),
    );
    glue::topology(list, 17);
    glue::draw(list, 6, 1, 3, 0);

    let recorded = with_list(list, |_, l| match l.list.cmds.last() {
        Some(crate::native::list::Cmd::Draw {
            topology,
            call:
                DrawCall::Arrays {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
            ..
        }) => Some((*topology, *vertex_count, *instance_count, *first_vertex, *first_instance)),
        _ => None,
    })
    .flatten()
    .expect("recorded draw");
    assert_eq!(recorded, (Topology::Triangles, 6, 1, 3, 0));
}

/// `CopyResource` between textures and `CopyTextureRegion` (subresource-to-subresource and
/// footprint-to/from-subresource) execute on Metal without dropping to `dropped_op`.
#[test]
fn copy_resource_and_copy_texture_region_run_on_the_gpu() {
    let Some(pool) = pool(9203, (512 << 10) / page_size() + 1) else { return };
    let tex_a = crate::resource::test_texture_at(
        pool.va(0),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let tex_b = crate::resource::test_texture_at(
        pool.va(0x10000),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let tex_c = crate::resource::test_texture_at(
        pool.va(0x20000),
        (16, 16),
        1,
        28,
        Swizzle::SW_64KB_R_X.0,
        1,
    );
    let up_buf = crate::resource::test_buffer_at(pool.va(0x30000), 0x4000);
    let down_buf = crate::resource::test_buffer_at(pool.va(0x40000), 0x4000);

    // Stage 8x8 linear texels at row pitch 256 into `up_buf`.
    let pitch = 256usize;
    let mut staged = vec![0u8; 7 * pitch + 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let at = y * pitch + x * 4;
            staged[at..at + 4].copy_from_slice(&[x as u8 * 11, y as u8 * 23, 77, 255]);
        }
    }
    pool.write(0x30000, &staged);

    let list = crate::list::test_list(false, std::ptr::null());
    let q = crate::queue::test_queue(false);
    let mut f = std::ptr::null();
    crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);

    // 1. CopyTextureRegion (footprint -> tex_a at (4, 4, 0) from 4x4 box starting at (2, 3))
    let src_fp = crate::list::CopyLocation::placed(
        up_buf,
        crate::resource::PlacedFootprint::new(0, 28, 8, 8, pitch as u32),
    );
    let dst_a = crate::list::CopyLocation::whole(tex_a);
    let box_a = [2u32, 3, 0, 6, 7, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_a, 4, 4, 0, &src_fp, &box_a);

    // 2. CopyResource (tex_a -> tex_b whole texture)
    slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, tex_b, tex_a);

    // 3. CopyTextureRegion (tex_b subresource (4, 4)..(8, 8) -> tex_c at (1, 2))
    let src_b = crate::list::CopyLocation::whole(tex_b);
    let dst_c = crate::list::CopyLocation::whole(tex_c);
    let box_b = [4u32, 4, 0, 8, 8, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_c, 1, 2, 0, &src_b, &box_b);

    // 4. CopyTextureRegion (tex_c subresource (1, 2)..(5, 6) -> down_buf footprint at (0, 0))
    let src_c = crate::list::CopyLocation::whole(tex_c);
    let dst_fp = crate::list::CopyLocation::placed(
        down_buf,
        crate::resource::PlacedFootprint::new(0, 28, 4, 4, pitch as u32),
    );
    let box_c = [1u32, 2, 0, 5, 6, 1];
    slot::<extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6])>(
        list, 0x80,
    )(list, &dst_fp, 0, 0, 0, &src_c, &box_c);

    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    let before_drops = crate::list::DROPPED_KINDS[3].load(std::sync::atomic::Ordering::Relaxed);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(
        slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(
            f,
            1,
            std::ptr::null_mut()
        ),
        0
    );
    let after_drops = crate::list::DROPPED_KINDS[3].load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(after_drops, before_drops, "texture copies must not be counted as dropped");

    let mut readback = vec![0u8; 3 * pitch + 16];
    assert!(crate::native::safe_read(pool.va(0x40000) as *const u8, &mut readback));
    for row in 0..4usize {
        for col in 0..4usize {
            let at = row * pitch + col * 4;
            assert_eq!(
                &readback[at..at + 4],
                &[(col as u8 + 2) * 11, (row as u8 + 3) * 23, 77, 255],
                "readback texel ({col}, {row})"
            );
        }
    }
}

#[test]
fn a_rectangular_color_clear_preserves_texels_outside_the_selected_regions() {
    for (storage_format, view_format, color, pixel) in [
        (28, 28, [1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        (28, 29, [0.5, 0.5, 0.5, 1.0], [188, 188, 188, 255]),
        (30, 30, [17.0, 65.0, 201.0, 255.0], [17, 65, 201, 255]),
    ] {
        check_rectangular_color_clear(storage_format, view_format, color, pixel);
    }
}

fn check_rectangular_color_clear(
    storage_format: u32,
    view_format: u32,
    color: [f32; 4],
    cleared_pixel: [u8; 4],
) {
    use crate::list::Rect;
    use victory_backend::backend::ComputeBackend;

    let pool = pool(9139, 16).expect("Metal device and generated texture pool");
    let heap = crate::descriptor::test_heap(false, 4);
    let handle = slot::<extern "system" fn(This) -> u64>(heap, 0x48)(heap);
    let resource = crate::resource::test_texture_array_at(
        pool.va(0), (8, 6), 2, 2, storage_format, Swizzle::LINEAR.0, 1,
    );
    let key = resource as usize;
    let desc = [view_format, 5, 1, 1, 1, 0];
    crate::descriptor::create_rtv(std::ptr::null(), resource, &desc, handle);

    let sentinel = |mip, slice| [17 + (mip * 40 + slice * 80) as u8, 31, 47, 255];
    let queue = crate::queue::test_queue(false);
    let mut fence = std::ptr::null();
    assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), 0);
    let cases: [(&str, &[[i32; 4]], &[usize]); 6] = [
        ("interior", &[[1, 1, 3, 2]], &[5, 6]),
        ("bounded", &[[-2, 1, 2, 2]], &[4, 5]),
        ("empty and inverted", &[[1, 1, 1, 2], [3, 1, 1, 2]], &[]),
        ("wholly outside", &[[4, 0, 5, 3]], &[]),
        ("overlapping and disjoint", &[[0, 0, 2, 1], [1, 0, 3, 1], [2, 2, 8, 3]], &[0, 1, 2, 10, 11]),
        ("zero count", &[], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ];

    for (case_index, (name, rects, required)) in cases.into_iter().enumerate() {
        with_title(|title| {
            let texture = title.texture_of(key).expect("realized texture");
            for slice in 0..2 {
                for mip in 0..2 {
                    let (w, h) = (8 >> mip, 6 >> mip);
                    let pixel = sentinel(mip, slice);
                    title.host.backend.upload_texture(&texture, mip, slice, &pixel.repeat(w * h), w * 4, 0);
                }
            }
        }).expect("Metal title");

        let list = crate::list::test_list(false, std::ptr::null());
        let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const Rect)>(list, 0x1a0);
        let unused = Rect { left: 0, top: 0, right: 0, bottom: 0 };
        let mut caller_rects: Vec<_> = rects.iter().map(|&[left, top, right, bottom]| {
            Rect { left, top, right, bottom }
        }).collect();
        let ptr = if caller_rects.is_empty() { &unused as *const _ } else { caller_rects.as_ptr() };
        clear(list, handle, &color, caller_rects.len() as u32, ptr);
        // The list must own the original coordinates at record time.
        caller_rects.fill(unused);
        std::hint::black_box(&caller_rects);

        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
        let value = case_index as u64 + 1;
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, value), 0);
        with_title(|title| {
            let fence = title.fence(fence as usize, 0).expect("native fence");
            assert!(title.host.backend.wait(&fence, value, 2000), "native clear did not finish");
            let texture = title.texture_of(key).expect("realized texture");
            for slice in 0..2 {
                for mip in 0..2 {
                    let (w, h) = (8 >> mip, 6 >> mip);
                    let mut data = vec![0; w * h * 4];
                    title.host.backend.read_texture(&texture, mip, slice, &mut data, w * 4, 0);
                    for (index, pixel) in data.as_chunks::<4>().0.iter().enumerate() {
                        let expected = if (mip, slice) == (1, 1) && required.contains(&index) {
                            cleared_pixel
                        } else {
                            sentinel(mip, slice)
                        };
                        assert_eq!(*pixel, expected, "DXGI {view_format}, {name}, mip {mip}, slice {slice}, pixel {index}");
                    }
                }
            }
        }).expect("Metal title");
        println!("DXGI {view_format}, {name}: {} changed / {} preserved in view, adjacent subresources unchanged", required.len(), 12 - required.len());
    }
}

/// Closes title list `list`, submits it and waits for the GPU to run it.
fn run_title_list(list: This) {
    let (q, f) = (crate::queue::test_queue(false), {
        let mut f = std::ptr::null();
        crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
        f
    });
    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()), 0);
}

type CopyRegion = extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6]);

/// `CopyResource` of two title textures is a GPU copy, and the destination stays the GPU's: a
/// later sync does not fill it from its title memory.
#[test]
fn copy_resource_copies_a_texture_on_the_gpu() {
    let Some(pool) = pool(9211, 8) else { return };
    let src = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    tile(&pool, 0, &crate::resource::texture_info(src).unwrap().layout);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, dst, src);
    run_title_list(list);
    let want: Vec<u8> = (0..16).flat_map(|y| (0..16).flat_map(move |x| texel(0, x, y))).collect();
    assert!(read_mip(dst as usize, 0, 16, 16) == want, "the copy did not reach the destination");
    // Its title memory changes (it never held the copy): the host texture keeps the copy.
    pool.write(65536, &[0x11u8; 1024]);
    with_title(|t| t.sync_textures());
    assert!(read_mip(dst as usize, 0, 16, 16) == want, "the destination was filled again from title memory");
}

/// Plain texels of a compressed block's size copy into compressed blocks (a GPU encoder's
/// output), which no view joins: through a buffer, one texel a block.
#[test]
fn copy_texture_region_moves_plain_texels_into_compressed_blocks() {
    let Some(pool) = pool(9213, 8) else { return };
    // DXGI 3 (R32G32B32A32_UINT) 2x2 into DXGI 77 (BC3) 8x8.
    let src = crate::resource::test_texture_at(pool.va(0), (2, 2), 1, 3, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (8, 8), 1, 77, Swizzle::LINEAR.0, 0);
    let blocks: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
    let info = crate::resource::texture_info(src).unwrap();
    let mut tiled = vec![0u8; info.layout.size as usize];
    info.layout.tile(&mut tiled, 0, 0, &blocks, 32, 64).unwrap();
    pool.write(0, &tiled);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<CopyRegion>(list, 0x80)(list, &crate::list::CopyLocation::whole(dst), 0, 0, 0, &crate::list::CopyLocation::whole(src), std::ptr::null());
    run_title_list(list);
    let out = with_title(|t| {
        let tex = t.texture_of(dst as usize).expect("a host texture");
        let mut out = vec![0u8; 64];
        t.host.backend.read_texture(&tex, 0, 0, &mut out, 32, 64);
        out
    })
    .unwrap();
    assert_eq!(out, blocks);
}

/// `ResolveSubresource` of single-sample textures: a copy of what both subresources hold (a
/// larger source is clipped to the destination).
#[test]
fn resolve_subresource_copies_what_both_hold() {
    let Some(pool) = pool(9214, 8) else { return };
    let src = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 28, Swizzle::LINEAR.0, 0);
    let dst = crate::resource::test_texture_at(pool.va(65536), (8, 8), 1, 28, Swizzle::LINEAR.0, 0);
    tile(&pool, 0, &crate::resource::texture_info(src).unwrap().layout);
    let list = crate::list::test_list(false, std::ptr::null());
    slot::<extern "system" fn(This, This, u32, This, u32, u32)>(list, 0x98)(list, dst, 0, src, 0, 28);
    run_title_list(list);
    let want: Vec<u8> = (0..8).flat_map(|y| (0..8).flat_map(move |x| texel(0, x, y))).collect();
    assert!(read_mip(dst as usize, 0, 8, 8) == want, "the resolve did not reach the destination");
}

/// A depth and stencil texture to and from buffer rows moves its 4-byte depth plane: rows in, then
/// a region of them out again into other rows.
#[test]
fn depth_stencil_rows_go_in_and_out_as_the_depth_plane() {
    use crate::list::CopyLocation;
    use crate::resource::PlacedFootprint;
    let Some(pool) = pool(9215, 8) else { return };
    // DXGI 19 (D32_FLOAT_S8X24_UINT), a depth target; rows as DXGI 41 (R32_FLOAT).
    let depth = crate::resource::test_texture_at(pool.va(0), (16, 16), 1, 19, Swizzle::LINEAR.0, 2);
    let buf = crate::resource::test_buffer_at(pool.va(65536), 65536);
    let values: Vec<f32> = (0..16 * 16).map(|i| i as f32 / 256.0).collect();
    for y in 0..16 {
        pool.write(65536 + y * 256, &values[y * 16..y * 16 + 16]);
    }
    let list = crate::list::test_list(false, std::ptr::null());
    let copy = slot::<CopyRegion>(list, 0x80);
    let rows = CopyLocation::placed(buf, PlacedFootprint::new(0, 41, 16, 16, 256));
    copy(list, &CopyLocation::whole(depth), 0, 0, 0, &rows, std::ptr::null());
    let out = CopyLocation::placed(buf, PlacedFootprint::new(8192, 41, 4, 2, 256));
    copy(list, &out, 0, 0, 0, &CopyLocation::whole(depth), &[3, 5, 0, 7, 7, 1]);
    run_title_list(list);
    for r in 0..2 {
        let got: Vec<f32> = pool.read(65536 + 8192 + r * 256, 4);
        let at = (5 + r) * 16 + 3;
        assert_eq!(got, values[at..at + 4], "row {r}");
    }
}

/// With `VICTORY_TEXTURE_DROP_MIPS`, GPU copies of a texture that left out its top mips go to the
/// host mip that holds the console's: a footprint copied into console mip 3 of a 1024x1024 texture
/// (host mip 1 with two left out) comes back out of it, and a copy into a dropped mip is skipped
/// and counted, not encoded at the wrong level.
#[test]
fn texture_copies_follow_the_mips_a_host_texture_left_out() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            let _ = with_title(|t| t.set_drop_mips(0));
        }
    }
    let Some(pool) = pool(9205, (9 << 20) / page_size() + 1) else { return };
    let tex = crate::resource::test_texture_at(pool.va(0), (1024, 1024), 11, 28, Swizzle::SW_64KB_S.0, 0);
    let up_buf = crate::resource::test_buffer_at(pool.va(0x80_0000), 0x4000);
    let down_buf = crate::resource::test_buffer_at(pool.va(0x81_0000), 0x4000);
    let pitch = 256usize;
    let mut staged = vec![0u8; 7 * pitch + 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let at = y * pitch + x * 4;
            staged[at..at + 4].copy_from_slice(&[x as u8 * 13, y as u8 * 29, 91, 255]);
        }
    }
    pool.write(0x80_0000, &staged);
    with_title(|t| t.set_drop_mips(2)).unwrap();
    let _reset = Reset;

    let list = crate::list::test_list(false, std::ptr::null());
    let q = crate::queue::test_queue(false);
    let mut f = std::ptr::null();
    crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f);
    type Region = extern "system" fn(This, *const crate::list::CopyLocation, u32, u32, u32, *const crate::list::CopyLocation, *const [u32; 6]);
    let src_fp = crate::list::CopyLocation::placed(up_buf, crate::resource::PlacedFootprint::new(0, 28, 8, 8, pitch as u32));
    let whole = [0u32, 0, 0, 8, 8, 1];
    // Into console mip 3 (128x128) at (16, 32), and into console mip 0, which is left out.
    let mip3 = crate::list::CopyLocation::subresource(tex, 3);
    slot::<Region>(list, 0x80)(list, &mip3, 16, 32, 0, &src_fp, &whole);
    let before = with_title(|t| t.stats.dropped_mip_copies).unwrap();
    let mip0 = crate::list::CopyLocation::subresource(tex, 0);
    slot::<Region>(list, 0x80)(list, &mip0, 0, 0, 0, &src_fp, &whole);
    assert_eq!(with_title(|t| t.stats.dropped_mip_copies).unwrap(), before + 1, "the copy into a dropped mip");
    assert_eq!(with_title(|t| (t.host_mip(tex as usize, 3), t.host_mip(tex as usize, 1))).unwrap(), (Some(1), None));
    // And back out of console mip 3.
    let dst_fp = crate::list::CopyLocation::placed(down_buf, crate::resource::PlacedFootprint::new(0, 28, 8, 8, pitch as u32));
    let back = [16u32, 32, 0, 24, 40, 1];
    slot::<Region>(list, 0x80)(list, &dst_fp, 0, 0, 0, &mip3, &back);

    assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), 0);
    slot::<extern "system" fn(This, u32, *const This)>(q, 0x50)(q, 1, &list);
    slot::<extern "system" fn(This, This, u64) -> i32>(q, 0x70)(q, f, 1);
    assert_eq!(slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> i32>(f, 0x48)(f, 1, std::ptr::null_mut()), 0);
    let mut readback = vec![0u8; staged.len()];
    assert!(crate::native::safe_read(pool.va(0x81_0000) as *const u8, &mut readback));
    for y in 0..8usize {
        for x in 0..8usize {
            let at = y * pitch + x * 4;
            assert_eq!(&readback[at..at + 4], &staged[at..at + 4], "texel ({x}, {y}) of console mip 3");
        }
    }
}
