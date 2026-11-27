//! Wave64 on a 32-lane host ([`Options::wave32_host`](crate::emit::Options)):
//! Apple's Metal Shader Converter takes `WaveSize(32)` only, and Apple GPUs
//! run 32-lane SIMD groups.
//!
//! One thread stays one console lane. A console wave of 64 lanes is two host
//! waves of the group (flat ids `64n..64n+31` and `64n+32..64n+63`), and a
//! thread's console lane is its group index modulo 64 (`console_flat`, which
//! also makes the host lane the console's). What stays inside 32 lanes is
//! the host's own: DPP rows and quads, `ds_swizzle` (its masks keep bit 5),
//! `ds_bpermute_b32`, `v_mbcnt`, `v_writelane`, lane-bit tests of masks held as numbers and
//! `v_readfirstlane` of a value the same in every lane of the console wave
//! ([`Lower::wave_uniform`]). Any other `v_readfirstlane` reads the first lane of
//! the console's 64-lane EXEC across the halves, or is refused (#254,
//! [`Lower::split_first_lane`]).
//!
//! What reaches across the two halves goes through group-shared memory (the
//! `xchg` array, one dword per thread) between two group barriers: a ballot
//! of a mask read as data, `v_readlane_b32`, `v_readfirstlane_b32` of a value
//! that differs between lanes, and the reductions and prefixes
//! of whole-wave sections. A ballot is each
//! half's 32 bits, joined; a lane read is a load at the source lane's slot;
//! a reduction is each half's result, combined; a prefix combines the lower
//! half's total with each upper lane's prefix. Lanes the group lacks (a group
//! whose lanes are not a multiple of 64) write no slot: their ballot bits are
//! 0, a missing upper half adds nothing to a reduction, and a lane read of
//! one gives 0.
//!
//! Group barriers need every thread of the group. Such an operation is
//! therefore lowered only in a block each thread runs exactly once
//! ([`once_blocks`](super::once_blocks)), or, in a group that is one console
//! wave, in one reached by uniform (SCC) branches alone and outside any loop
//! ([`wave_blocks`](super::wave_blocks)): both halves then arrive together.
//! An SCC computed from a `v_readfirstlane` is not uniform here: each half
//! reads its own first lane. Elsewhere (in a branch of EXEC, a
//! per-lane loop) the translation is refused. Cost: two barriers and a
//! shared-memory round trip per such operation, 4 bytes per thread of
//! group-shared memory in all, and `WaveSize(32)` in place of 64.
//!
//! LDS that lanes of one console wave share without a barrier (the console
//! runs the wave in lockstep) gets the group barriers `narrow_waves` adds
//! ([`split_wave_barriers`](super::split_wave_barriers)), or is refused.

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::shader::flags;
use crate::emit::wave::Op;

impl Lower<'_> {
    /// A group barrier with a shared-memory fence in DXIL block `at`.
    fn xchg_barrier(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Whether every thread of the group (both halves of a console wave)
    /// reaches IR block `ir` together.
    fn exchange_site(&self, ir: usize) -> Result<(), Why> {
        // In lockstep every thread reaches every block together (`lockstep.rs`).
        if self.once[ir] || self.x.st.lockstep {
            return Ok(());
        }
        Err(super::DIVERGENT_EXCHANGE)
    }

    /// The group-shared slot `index`.
    fn xchg_slot(&mut self, at: usize, index: V) -> V {
        let g = self.s.xchg();
        self.s.m.element(at, g, index)
    }

    /// Whether exchange slot `index`, which holds lanes `index << shift` on, is a thread's, where
    /// the group's lanes are not a multiple of 64 (its last console wave lacks lanes, or a whole
    /// host wave, and no thread writes their slots). `None`: every slot is a thread's.
    fn xchg_written(&mut self, at: usize, index: V, shift: u32) -> Option<V> {
        let lanes: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        if lanes.is_multiple_of(64) {
            return None;
        }
        let k = self.s.m.i32(shift);
        let first = self.s.m.bin(at, Bin::Shl, index, k);
        let n = self.s.m.i32(lanes);
        Some(self.s.m.cmp(at, pred::IULT, first, n))
    }

    /// The low and high words of the current host wave's active ballot.
    pub(super) fn host_ballot(&mut self, at: usize, on: V) -> (V, V) {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        self.s.flags |= flags::WAVE_OPS;
        (
            self.s.m.extract(at, ballot, 0),
            self.s.m.extract(at, ballot, 1),
        )
    }

    /// The console's ballot of `on` in DXIL block `at` (IR block `ir`): the
    /// low and high dwords.
    pub(super) fn wide_ballot(&mut self, at: usize, ir: usize, on: V) -> Result<(V, V), Why> {
        let (x, y) = self.host_ballot(at, on);
        if !self.halves {
            return Ok((x, y));
        }
        self.exchange_site(ir)?;
        let f = self.console_flat();
        let (five, six, one) = (self.s.m.i32(5), self.s.m.i32(6), self.s.m.i32(1));
        let mine = self.s.m.bin(at, Bin::LShr, f, five);
        let slot = self.xchg_slot(at, mine);
        self.s.m.store(at, slot, x);
        self.xchg_barrier(at);
        let pair = self.s.m.bin(at, Bin::LShr, f, six);
        let lo_at = self.s.m.bin(at, Bin::Shl, pair, one);
        let hi_at = self.s.m.bin(at, Bin::Add, lo_at, one);
        let (lo_p, hi_p) = (self.xchg_slot(at, lo_at), self.xchg_slot(at, hi_at));
        let (lo, hi) = (self.s.m.load(at, lo_p), self.s.m.load(at, hi_p));
        self.xchg_barrier(at);
        // A missing upper half has no lanes on.
        let hi = match self.xchg_written(at, hi_at, 5) {
            Some(there) => {
                let z = self.s.m.i32(0);
                self.s.m.select(at, there, hi, z)
            }
            None => hi,
        };
        Ok((lo, hi))
    }

    /// `x` of console lane `lane` (the low six bits count).
    pub(super) fn wide_read_lane(&mut self, x: V, lane: V) -> Result<V, Why> {
        if !self.host32 {
            return Ok(self.read_lane(x, lane));
        }
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        if !self.halves {
            // A group of 32 lanes or fewer: no thread holds a lane above 31.
            self.s.flags |= flags::WAVE_OPS;
            let low = self.s.m.bin(c, Bin::And, lane, k31);
            let r = self.read_lane(x, low);
            let there = self.s.m.cmp(c, pred::IULT, lane, k32);
            return Ok(self.s.m.select(c, there, r, z));
        }
        self.exchange_site(self.cur_ir)?;
        self.s.flags |= flags::WAVE_OPS;
        let f = self.console_flat();
        let slot = self.xchg_slot(c, f);
        self.s.m.store(c, slot, x);
        self.xchg_barrier(c);
        let (k63, not63) = (self.s.m.i32(63), self.s.m.i32(!63));
        let wave = self.s.m.bin(c, Bin::And, f, not63);
        let lane = self.s.m.bin(c, Bin::And, lane, k63);
        let src = self.s.m.bin(c, Bin::Or, wave, lane);
        let p = self.xchg_slot(c, src);
        let r = self.s.m.load(c, p);
        self.xchg_barrier(c);
        // Guess: a lane the group lacks reads as 0 (the console reads whatever its VGPR holds).
        Ok(match self.xchg_written(c, src, 0) {
            Some(there) => self.s.m.select(c, there, r, z),
            None => r,
        })
    }

    /// `v_readfirstlane_b32` of a console wave split into two host waves (#254). The console reads
    /// the first lane of its 64-lane EXEC, and every lane sees that value; `waveReadLaneFirst`
    /// reads the first lane of each host wave, its own half.
    ///
    /// - A value the same in every lane of the console wave ([`Self::wave_uniform`]) is read per
    ///   half as before: each half's first lane holds it.
    /// - Otherwise, where both halves arrive together ([`Self::exchange_site`]), the lane is the
    ///   lowest bit of EXEC as `inst` reads it, read across the halves ([`Self::wide_read_lane`]).
    /// - Otherwise the program is refused; a group of one console wave is then tried in lockstep,
    ///   which reads it exactly.
    pub(super) fn split_first_lane(&mut self, inst: &crate::ir::Inst, a: V) -> Result<Option<V>, Why> {
        use crate::gfx10::{Fields, Operand};
        use crate::ir::Reg;
        if !self.host32 || !self.halves {
            return Ok(None);
        }
        let source = match &inst.fields {
            Fields::Valu(f) => match f.src.first() {
                Some(&Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)).map(|v| (Reg::V(reg), v)),
                Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)).map(|v| (Reg::S(code), v)),
                // A constant is the same in every lane.
                Some(Operand::Const(_) | Operand::Literal(_)) => return Ok(None),
                _ => None,
            },
            _ => None,
        };
        if let Some((reg, v)) = source
            && self.wave_uniform(inst, reg, v)
        {
            return Ok(None);
        }
        if self.exchange_site(self.cur_ir).is_err() {
            return Err("a first lane read across the two host waves of a console wave, in divergent code");
        }
        let lane = self.lockstep_first_lane(inst)?;
        Ok(Some(self.wide_read_lane(a, lane)?))
    }

    /// Whether `v`, what `reg` holds where `at` reads it, is provably the same in every lane that
    /// is on in `at`'s EXEC, in both halves of the console wave. Conservative: what it cannot show
    /// is not uniform. Uniform are constants; program inputs in SGPRs (user data); scalar
    /// instructions of uniform values that read no lane mask; first-lane reads (exact since #254,
    /// waterfall reads aside); values an earlier whole-wave section left the same in every lane;
    /// and VALU instructions of uniform values, made under `at`'s own EXEC or under the program's
    /// entry EXEC (which holds every lane that is ever on), with no DPP and no lane mask read.
    /// guess: the entry EXEC of a compute wave holds every lane any later EXEC turns on.
    pub(super) fn wave_uniform(&self, at: &crate::ir::Inst, reg: crate::ir::Reg, v: crate::ir::Value) -> bool {
        use crate::ir::{EXEC, Reg};
        let execs = (at.read(Reg::S(EXEC)), at.read(Reg::S(EXEC + 1)));
        let mut seen = std::collections::HashSet::new();
        self.uniform_in(execs, reg, v, &mut seen, 0)
    }

    fn uniform_in(
        &self,
        execs: (Option<crate::ir::Value>, Option<crate::ir::Value>),
        reg: crate::ir::Reg,
        v: crate::ir::Value,
        seen: &mut std::collections::HashSet<crate::ir::Value>,
        depth: u32,
    ) -> bool {
        use crate::gfx10::{Fields, Operand};
        use crate::ir::{EXEC, Reg, VCC};
        if depth > 48 {
            return false;
        }
        if !seen.insert(v) {
            // Only phis close a cycle, and they are uniform below only when every other
            // argument is this one value.
            return true;
        }
        if self.section_uniform(v) {
            return true;
        }
        let mask = |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1 || c == VCC || c == VCC + 1);
        if mask(reg) {
            // A lane mask read as a number: with the halves on it is the console's 64 bits in
            // both (a ballot through `xchg`), so the same in every lane.
            return self.halves;
        }
        let p = self.x.p;
        let Some(inst) = self.def_inst(v) else {
            let Some(&b) = self.def_block.get(&v) else {
                // A program input: user data in SGPRs is the wave's; VGPR inputs are per lane.
                return matches!(reg, Reg::S(_));
            };
            let Some(phi) = p.blocks[b].phis.iter().find(|ph| ph.value == v) else {
                return false;
            };
            // A scalar join is reached by scalar branches, which the whole console wave takes
            // together (a first lane read is the wave's since #254): uniform when every edge
            // brings a uniform value. A vector join merges lanes from different paths: uniform
            // only when every edge brings the same value.
            let args: Vec<_> = phi.args.iter().copied().filter(|&x| x != v).collect();
            let Some(&first) = args.first() else {
                return false;
            };
            if matches!(phi.reg, Reg::S(_)) {
                return args.iter().all(|&x| self.uniform_in(execs, phi.reg, x, seen, depth + 1));
            }
            return args.iter().all(|&x| x == first) && self.uniform_in(execs, phi.reg, first, seen, depth + 1);
        };
        let name = inst.name();
        // `vector`: a VALU instruction, which reads a lane mask per lane (a lane's own bit), so a
        // mask operand is not uniform there; its EXEC is checked apart, and so is its read of its
        // destination's old value, which only the lanes outside EXEC keep (and `at` reads none of
        // those: its EXEC is the same, or within the entry EXEC).
        let dst = match &inst.fields {
            Fields::Valu(f) => match f.dst {
                Some(Operand::Vector { reg, dwords }) => Some((reg, dwords.max(1))),
                _ => None,
            },
            _ => None,
        };
        let old = |r: Reg| matches!((r, dst), (Reg::V(x), Some((d, n))) if x >= d && x < d.saturating_add(n));
        let reads_ok = |l: &Self, seen: &mut std::collections::HashSet<crate::ir::Value>, vector: bool| {
            inst.reads.iter().all(|&(r, x)| {
                if vector && (matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) || old(r)) {
                    return true;
                }
                !(vector && mask(r)) && l.uniform_in(execs, r, x, seen, depth + 1)
            })
        };
        match &inst.fields {
            // A first-lane read is the console wave's first lane since #254 (exact, or refused);
            // a waterfall loop's read is the lane's own.
            Fields::Valu(_) if name == "v_readfirstlane_b32" => {
                !inst.writes.iter().any(|w| self.waterfall_reads.contains(&w.1))
            }
            Fields::Valu(f) => {
                if f.dpp.is_some()
                    || name.starts_with("v_mbcnt")
                    || name.starts_with("v_cmp")
                    || name.starts_with("v_readlane")
                    || name.starts_with("v_writelane")
                    || name.starts_with("v_interp")
                    || f.src.iter().any(|o| matches!(o, Operand::Attr { .. }))
                {
                    return false;
                }
                // Made where at least `at`'s lanes are on: the same EXEC, or the entry EXEC.
                let def = (inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
                let entry = |x: Option<crate::ir::Value>| {
                    x.is_some_and(|x| self.def_inst(x).is_none() && !self.def_block.contains_key(&x))
                };
                if def != execs && !(entry(def.0) && entry(def.1)) {
                    return false;
                }
                reads_ok(self, seen, true)
            }
            // Scalar ALU and scalar loads of uniform operands; anything reading a lane mask as a
            // number is per half.
            Fields::Sop { .. } | Fields::Smem { .. } => reads_ok(self, seen, false),
            _ => false,
        }
    }

    /// The console lane `v_readfirstlane_b32` reads in lockstep (`lockstep.rs`): the lowest bit of
    /// EXEC as `inst` reads it, as a number (0 when none is set).
    pub(super) fn lockstep_first_lane(&mut self, inst: &crate::ir::Inst) -> Result<V, Why> {
        use crate::ir::{EXEC, Reg};
        let c = self.cur;
        let word = |l: &mut Self, r: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(r)).ok_or("EXEC of a first lane read")?;
            let was = std::mem::replace(&mut l.mask_ok, true);
            let x = l.i32_of(v);
            l.mask_ok = was;
            x
        };
        let (lo, hi) = (word(self, EXEC)?, word(self, EXEC + 1)?);
        let i32t = self.s.t.i32;
        let flo = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let (z, k32, k63) = (self.s.m.i32(0), self.s.m.i32(32), self.s.m.i32(63));
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let low = self.s.m.cmp(c, pred::NE, lo, z);
        let r = self.s.m.select(c, low, flo, fhi);
        Ok(self.s.m.bin(c, Bin::And, r, k63))
    }

    /// Extend a host wave's result over the console's 64 lanes. A prefix
    /// also supplies its combining operation and the whole host wave's total.
    pub(super) fn halves_reduce(
        &mut self,
        op: Op,
        r: V,
        prefix: Option<(Bin, V)>,
    ) -> Result<V, Why> {
        let c = self.cur;
        self.exchange_site(self.cur_ir)?;
        let float = matches!(op, Op::AddF | Op::MulF | Op::MinF | Op::MaxF);
        let mine = prefix.map_or(r, |(_, total)| total);
        let bits = if float { self.as_int(mine) } else { mine };
        let f = self.console_flat();
        let (five, six, one) = (self.s.m.i32(5), self.s.m.i32(6), self.s.m.i32(1));
        let half = self.s.m.bin(c, Bin::LShr, f, five);
        let slot = self.xchg_slot(c, half);
        self.s.m.store(c, slot, bits);
        self.xchg_barrier(c);
        let pair = self.s.m.bin(c, Bin::LShr, f, six);
        let lo_at = self.s.m.bin(c, Bin::Shl, pair, one);
        let hi_at = self.s.m.bin(c, Bin::Add, lo_at, one);
        let (lo_p, hi_p) = (self.xchg_slot(c, lo_at), self.xchg_slot(c, hi_at));
        let (lo, hi) = (self.s.m.load(c, lo_p), self.s.m.load(c, hi_p));
        self.xchg_barrier(c);
        let (lo, hi) = if float {
            (self.as_float(lo), self.as_float(hi))
        } else {
            (lo, hi)
        };
        if let Some((bin, _)) = prefix {
            // The upper half's lanes also have every lane of the lower below.
            let upper = self.s.m.bin(c, Bin::And, half, one);
            let z = self.s.m.i32(0);
            let is_upper = self.s.m.cmp(c, pred::NE, upper, z);
            let combined = self.s.m.bin(c, bin, lo, r);
            return Ok(self.s.m.select(c, is_upper, combined, r));
        }
        let both = match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::MulF => self.s.m.bin(c, Bin::Mul, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinF | Op::MaxF => self.max_min_flushed(op == Op::MaxF, lo, hi),
            Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                let p = match op {
                    Op::MinU => pred::IULT,
                    Op::MaxU => pred::IUGT,
                    Op::MinI => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, lo, hi);
                self.s.m.select(c, t, lo, hi)
            }
            Op::AddH => return Err("a 16-bit wave reduction across the host's waves"),
        };
        // A missing upper half adds nothing: the lower half's result alone.
        Ok(match self.xchg_written(c, hi_at, 5) {
            Some(there) => self.s.m.select(c, there, both, lo),
            None => both,
        })
    }
}
