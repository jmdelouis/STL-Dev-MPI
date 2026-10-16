//! In-process backend: one thread per rank, messages over channels.  Used
//! to test the decomposition and the exchanges without an MPI launcher.

use crate::comm::Comm;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};

enum Payload {
    F64(Vec<f64>),
    I64(Vec<i64>),
}

/// Message: (collective sequence number, sender, payload).
type Message = (u64, usize, Payload);

pub struct ThreadComm {
    rank: usize,
    size: usize,
    peers: Vec<Sender<Message>>,
    inbox: Receiver<Message>,
    seq: Cell<u64>,
    early: RefCell<HashMap<(u64, usize), Payload>>,
}

impl ThreadComm {
    fn exchange(&self, mut send: Vec<Payload>) -> Vec<Payload> {
        assert_eq!(send.len(), self.size);
        let seq = self.seq.get();
        self.seq.set(seq + 1);
        let own = std::mem::replace(&mut send[self.rank], Payload::I64(Vec::new()));
        for (peer, payload) in send.into_iter().enumerate() {
            if peer != self.rank {
                self.peers[peer].send((seq, self.rank, payload)).expect("peer thread ended");
            }
        }
        let mut got: Vec<Option<Payload>> = (0..self.size).map(|_| None).collect();
        got[self.rank] = Some(own);
        let mut early = self.early.borrow_mut();
        for (peer, slot) in got.iter_mut().enumerate() {
            if let Some(p) = early.remove(&(seq, peer)) {
                *slot = Some(p);
            }
        }
        while got.iter().any(Option::is_none) {
            let (s, from, payload) = self.inbox.recv().expect("peer thread ended");
            if s == seq {
                got[from] = Some(payload);
            } else {
                // A faster peer is already in a later collective.
                early.insert((s, from), payload);
            }
        }
        got.into_iter().map(Option::unwrap).collect()
    }
}

impl Comm for ThreadComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn alltoallv_f64(&self, send: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.exchange(send.into_iter().map(Payload::F64).collect())
            .into_iter()
            .map(|p| match p {
                Payload::F64(v) => v,
                Payload::I64(_) => panic!("collectives out of order"),
            })
            .collect()
    }

    fn alltoallv_i64(&self, send: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
        self.exchange(send.into_iter().map(Payload::I64).collect())
            .into_iter()
            .map(|p| match p {
                Payload::I64(v) => v,
                Payload::F64(_) => panic!("collectives out of order"),
            })
            .collect()
    }
}

/// Run `f` on `size` ranks (one thread each); returns the results in rank order.
pub fn run_ranks<T: Send>(size: usize, f: impl Fn(&ThreadComm) -> T + Sync) -> Vec<T> {
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..size).map(|_| channel::<Message>()).unzip();
    std::thread::scope(|scope| {
        let handles: Vec<_> = receivers
            .into_iter()
            .enumerate()
            .map(|(rank, inbox)| {
                let peers = senders.clone();
                let f = &f;
                scope.spawn(move || {
                    let comm = ThreadComm {
                        rank,
                        size,
                        peers,
                        inbox,
                        seq: Cell::new(0),
                        early: RefCell::new(HashMap::new()),
                    };
                    f(&comm)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("rank panicked")).collect()
    })
}
