use {
    ouroboros::self_referencing,
    std::{
        iter::{Cycle, Take},
        slice,
    },
};

/// Creates a cycle iterator over a slice of `Item`s that reside in the `Holder`.
#[self_referencing]
pub(crate) struct Cycler<Holder: 'static, Item: 'static> {
    /// Holds the accounts that are referenced by the `iterator`.
    holder: Holder,

    #[borrows(holder)]
    #[covariant]
    iterator: Cycle<slice::Iter<'this, Item>>,
}

impl<Holder, Item> Cycler<Holder, Item> {
    /// Constructs a `Cycler` that holds `source` and will iterate over `Item`s selected via the
    /// `selector` projection function.
    ///
    /// For example, if `source` is a vector of values, `selector` can return an iterator over a
    /// subslice of those.
    ///
    /// `selector` must not return an empty iterator, or `take_one()` will panic.
    pub(crate) fn over<Selector>(source: Holder, selector: Selector) -> Self
    where
        Selector: for<'holder> FnOnce(&'holder Holder) -> slice::Iter<'holder, Item>,
    {
        CyclerBuilder {
            holder: source,
            iterator_builder: |holder| selector(holder).cycle(),
        }
        .build()
    }

    /// Runs `f`, passing in an iterator that goes over `chunk_size` `Item`s.  Internal state is a
    /// cycle iterator, so calling this method multiple times will continue using `Item`s that were
    /// not used in the previous call.  And if it runs out of `Item`s it will restart from the
    /// beginning.
    pub(crate) fn with_chunk<'holder, F, Res>(&'holder mut self, chunk_size: usize, f: F) -> Res
    where
        F: for<'this> FnOnce(ChunkIter<'holder, 'this, Item>) -> Res,
    {
        self.with_iterator_mut(|iterator| {
            f(ChunkIter {
                inner: iterator.take(chunk_size),
                len: chunk_size,
            })
        })
    }

    /// Returns a reference to a single `Item`, advancing internal iterator by one.
    pub(crate) fn take_one(&mut self) -> &Item {
        self.with_iterator_mut(|iterator| iterator.next().expect("Cycle iterator never ends"))
    }
}

/// An iterator over a chunk of `Item`s in the [`Cycler`], used by [`Cycler::with_chunk()`].
///
/// It is, essentially, an `ExactSizeIterator<Item = &'holder Item> + 'iter`.
pub(crate) struct ChunkIter<'holder, 'this, Item> {
    inner: Take<&'holder mut Cycle<slice::Iter<'this, Item>>>,

    // `Take` does not implementation `ExactSizeIterator` when the referenced iterator does not.
    // And `Cycle` does not implementation `ExactSizeIterator` as it is infinite.  But we do want to
    // be able to call `len()` on the chunk provided to the `f` callback in
    // [`Cycler::with_chunk()`].
    len: usize,
}

impl<'holder, Item> Iterator for ChunkIter<'holder, '_, Item> {
    type Item = &'holder Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<Item> ExactSizeIterator for ChunkIter<'_, '_, Item> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use {
        super::Cycler,
        std::{iter::zip, rc::Rc},
    };

    #[test]
    fn borrow_from_vec() {
        let data = vec![1, 2, 3, 4];

        let mut cycle = Cycler::over(data, |data| data[1..=2].iter());

        let actual = cycle.with_chunk(5, |it| it.copied().collect::<Vec<_>>());
        assert_eq!(actual, vec![2, 3, 2, 3, 2]);
    }

    #[test]
    fn borrow_from_shared_vec() {
        let data = Rc::new(vec![1, 2, 3, 4]);

        let mut cycle1 = Cycler::over(data.clone(), |data| data[1..=2].iter());
        let mut cycle2 = Cycler::over(data.clone(), |data| data[0..=2].iter());

        let actual = cycle1.with_chunk(5, |it1| {
            cycle2.with_chunk(5, |it2| zip(it1.copied(), it2.copied()).collect::<Vec<_>>())
        });

        assert_eq!(actual, vec![(2, 1), (3, 2), (2, 3), (3, 1), (2, 2)]);
    }

    #[test]
    fn take_one() {
        let data = vec![1, 2, 3, 4];

        let mut cycle = Cycler::over(data, |data| data[1..=3].iter());

        let actual = (0..7).map(|_| *cycle.take_one()).collect::<Vec<_>>();
        assert_eq!(actual, vec![2, 3, 4, 2, 3, 4, 2]);
    }

    #[test]
    fn chunk_size() {
        let data = vec![1, 2, 3, 4];

        let mut cycle = Cycler::over(data, |data| data[1..=2].iter());

        let actual = cycle.with_chunk(5, |it| it.len());
        assert_eq!(actual, 5);
    }
}
