//! Tiny owned GPU fixture helpers; no model or timing study.
use super::*;
pub(super) struct Guarded {
    pub(super) parent: MlxBuffer,
    pub(super) view: MlxBuffer,
    pub(super) dtype: DType,
    pub(super) count: usize,
}
impl Guarded {
    pub(super) fn new(d: &MlxDevice, n: usize, dtype: DType) -> Self {
        let mut parent = d.alloc_buffer((n + 8) * 4, dtype, vec![n + 8]).unwrap();
        match dtype {
            DType::F32 => parent
                .as_mut_slice::<f32>()
                .unwrap()
                .fill(f32::from_bits(0x7fc12345)),
            DType::U32 => parent.as_mut_slice::<u32>().unwrap().fill(0xa5a5a5a5),
            DType::I32 => parent.as_mut_slice::<i32>().unwrap().fill(-1234567),
            _ => unreachable!(),
        }
        let view = parent.slice_view(16, n);
        Self {
            parent,
            view,
            dtype,
            count: n,
        }
    }
    pub(super) fn guards(&self) {
        let intact = match self.dtype {
            DType::F32 => {
                let x = self.parent.as_slice::<f32>().unwrap();
                x[..4]
                    .iter()
                    .chain(&x[self.count + 4..])
                    .all(|v| v.to_bits() == 0x7fc12345)
            }
            DType::U32 => {
                let x = self.parent.as_slice::<u32>().unwrap();
                x[..4]
                    .iter()
                    .chain(&x[self.count + 4..])
                    .all(|&v| v == 0xa5a5a5a5)
            }
            DType::I32 => {
                let x = self.parent.as_slice::<i32>().unwrap();
                x[..4]
                    .iter()
                    .chain(&x[self.count + 4..])
                    .all(|&v| v == -1234567)
            }
            _ => false,
        };
        assert!(intact, "logical view guard changed");
    }
    pub(super) fn ids(&self) -> Vec<u32> {
        match self.dtype {
            DType::U32 => self.view.as_logical_slice::<u32>().unwrap().to_vec(),
            DType::I32 => self
                .view
                .as_logical_slice::<i32>()
                .unwrap()
                .iter()
                .map(|&x| x as u32)
                .collect(),
            _ => unreachable!(),
        }
    }
    pub(super) fn bits(&self) -> Vec<u32> {
        self.view
            .as_logical_slice::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }
}
pub(super) fn input(d: &MlxDevice, v: &[f32]) -> MlxBuffer {
    let mut b = d
        .alloc_buffer((v.len() + 8) * 4, DType::F32, vec![v.len() + 8])
        .unwrap();
    let data = b.as_mut_slice::<f32>().unwrap();
    data.fill(f32::NAN);
    data[4..4 + v.len()].copy_from_slice(v);
    b.slice_view(16, v.len())
}
pub(super) struct Outputs {
    pub(super) ids: Guarded,
    pub(super) weights: Guarded,
    pub(super) scores: Guarded,
    pub(super) boundary_ids: Guarded,
    pub(super) status: Guarded,
}
impl Outputs {
    pub(super) fn new(d: &MlxDevice, rows: usize, k: usize, idtype: DType) -> Self {
        Self {
            ids: Guarded::new(d, rows * k, idtype),
            weights: Guarded::new(d, rows * k, DType::F32),
            scores: Guarded::new(d, rows * 2, DType::F32),
            boundary_ids: Guarded::new(d, rows * 2, DType::U32),
            status: Guarded::new(d, rows, DType::U32),
        }
    }
    pub(super) fn witness(&self) -> SelectionWitness<'_> {
        SelectionWitness {
            scores: &self.scores.view,
            ids: &self.boundary_ids.view,
            status: &self.status.view,
        }
    }
    pub(super) fn check(&self) {
        for g in [
            &self.ids,
            &self.weights,
            &self.scores,
            &self.boundary_ids,
            &self.status,
        ] {
            g.guards();
        }
    }
}
pub(super) fn parity(old: &Outputs, new: &Outputs) {
    assert_eq!(old.ids.ids(), new.ids.ids(), "selected IDs/order changed");
    assert_eq!(
        old.weights.bits(),
        new.weights.bits(),
        "selected weight bits changed"
    );
    assert!(new
        .weights
        .view
        .as_logical_slice::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    old.check();
    new.check();
}
pub(super) fn assert_selected_boundary(o: &Outputs, rows: u32, k: u32, experts: u32) {
    let selected = o.ids.ids();
    for (row, b) in read_after_checked_drain(o.witness(), rows, experts, k)
        .unwrap()
        .iter()
        .enumerate()
    {
        match b {
            SelectionBoundary::Scored {
                kth_id, next_id, ..
            } => {
                assert_eq!(*kth_id, selected[row * k as usize + k as usize - 1]);
                assert!(!selected[row * k as usize..(row + 1) * k as usize].contains(next_id));
            }
            SelectionBoundary::AllExpertsSelected { kth_id, .. } => {
                assert_eq!(*kth_id, selected[row * k as usize + k as usize - 1])
            }
            SelectionBoundary::Invalid => panic!("valid fixture rejected"),
        }
    }
}

pub(super) fn replay(enc: &mut CommandEncoder, inputs: &[&MlxBuffer], o: &Outputs) {
    use crate::{encoder::CapturedNode, graph::ComputeGraph};
    let nodes = enc.take_capture().expect("actual witness capture");
    assert_eq!(nodes.len(), 1);
    let expected_reads = inputs.iter().map(|b| range(b).unwrap()).collect::<Vec<_>>();
    let expected_writes = [
        &o.ids.view,
        &o.weights.view,
        &o.scores.view,
        &o.boundary_ids.view,
        &o.status.view,
    ]
    .iter()
    .map(|b| range(b).unwrap())
    .collect::<Vec<_>>();
    match &nodes[0] {
        CapturedNode::Dispatch { reads, writes, .. } => {
            assert_eq!(*reads, expected_reads);
            assert_eq!(*writes, expected_writes);
        }
        _ => panic!("expected one actual witness producer"),
    }
    let mut graph = ComputeGraph::from_nodes(nodes);
    assert_eq!(graph.unannotated_dispatch_count(), 0);
    graph.reorder();
    graph.encode_with_barriers(enc);
}
