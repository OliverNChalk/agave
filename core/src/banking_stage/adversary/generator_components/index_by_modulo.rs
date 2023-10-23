/// A simple helper to choose indices in a loop, limited by a maximum value, at which point the
/// index loops back to 0.
///
/// Almost identical to `(0..next).cycle()` except that `next()` returns `usize` rather than
/// `Option<usize>`.
pub struct IndexByModulo {
    // Next index to return.
    next: usize,

    // Wrap when we reach this value.
    max: usize,
}

impl IndexByModulo {
    pub fn new(max: usize) -> Self {
        assert!(
            max > 0,
            "There should be at least one valid index.  Got `max` of 0."
        );
        Self { next: 0, max }
    }

    pub fn next(&mut self) -> usize {
        let Self { next, max } = *self;
        self.next = (next + 1) % max;
        next
    }
}

#[cfg(test)]
mod tests {
    use super::IndexByModulo;

    #[test]
    fn loops_around() {
        let mut index = IndexByModulo::new(4);

        let actual = (0..=9).map(|_| index.next()).collect::<Vec<_>>();
        assert_eq!(actual, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]);
    }
}
